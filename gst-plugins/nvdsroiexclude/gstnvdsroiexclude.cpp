#include <gst/base/gstbasetransform.h>
#include <gst/gst.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

#include "gstnvdsmeta.h"
#include "nvdsmeta.h"

#ifndef PACKAGE
#define PACKAGE "noesis"
#endif

GST_DEBUG_CATEGORY_STATIC(gst_nvds_roi_exclude_debug);
#define GST_CAT_DEFAULT gst_nvds_roi_exclude_debug

namespace {

constexpr size_t kMaxConfigBytes = 1024 * 1024;
constexpr int kMaxConfigDimension = 16384;

enum PropertyId : guint {
  kPropertyNone = 0,
  kPropertyConfigFile,
  kPropertyIdMode,
  kPropertyReloadRequestSequence,
  kPropertyReloadAcceptedSequence,
  kPropertyReloadFailedSequence,
  kPropertyLastReloadOk,
  kPropertyExpectedConfigSha256,
  kPropertyActiveConfigSha256,
  kPropertyReloadErrorCount,
  kPropertyObjectsRemovedCount,
  kPropertyLastReloadError,
  kPropertyCount,
};

GParamSpec *properties[kPropertyCount] = {};

struct Point {
  double x = 0.0;
  double y = 0.0;
};

struct Polygon {
  std::string label;
  std::vector<Point> points;
  bool enabled = true;
};

struct StreamConfig {
  bool enabled = true;
  int class_id = -1;
  bool inverse_roi = false;
  int config_width = 0;
  int config_height = 0;
  std::vector<Polygon> polygons;
};

struct RuntimeConfig {
  bool enabled = true;
  int osd_mode = 0;
  int display_font_size = 4;
  int config_width = 1920;
  int config_height = 1080;
  std::map<int, StreamConfig> streams;
};

struct MappedPolygon {
  std::string label;
  std::vector<Point> points;
};

static bool parse_bool(GKeyFile *key_file, const char *group, const char *key,
                       bool fallback, bool *out_value, std::string *out_error) {
  if (!g_key_file_has_key(key_file, group, key, nullptr)) {
    *out_value = fallback;
    return true;
  }
  GError *error = nullptr;
  gboolean value = g_key_file_get_boolean(key_file, group, key, &error);
  if (error != nullptr) {
    *out_error = std::string("invalid boolean ") + group + "." + key + ": " + error->message;
    g_error_free(error);
    return false;
  }
  *out_value = value != FALSE;
  return true;
}

static bool parse_int(GKeyFile *key_file, const char *group, const char *key,
                      int fallback, int *out_value, std::string *out_error) {
  if (!g_key_file_has_key(key_file, group, key, nullptr)) {
    *out_value = fallback;
    return true;
  }
  GError *error = nullptr;
  int value = g_key_file_get_integer(key_file, group, key, &error);
  if (error != nullptr) {
    *out_error = std::string("invalid integer ") + group + "." + key + ": " + error->message;
    g_error_free(error);
    return false;
  }
  *out_value = value;
  return true;
}

static bool valid_label(const char *value) {
  if (value == nullptr || *value == '\0') {
    return false;
  }
  const size_t length = std::strlen(value);
  if (length > 64 || !g_ascii_isalnum(value[0])) {
    return false;
  }
  for (size_t index = 1; index < length; ++index) {
    const unsigned char ch = static_cast<unsigned char>(value[index]);
    if (!g_ascii_isalnum(ch) && ch != '_' && ch != '-' && ch != '.') {
      return false;
    }
  }
  return true;
}

static bool parse_points(const char *raw, int width, int height,
                         std::vector<Point> *out_points,
                         std::string *out_error) {
  std::vector<double> values;
  if (raw == nullptr) {
    *out_error = "ROI coordinate list is missing";
    return false;
  }

  gchar **tokens = g_strsplit(raw, ";", -1);
  for (gchar **token = tokens; token != nullptr && *token != nullptr; ++token) {
    gchar *stripped = g_strstrip(*token);
    if (stripped == nullptr || *stripped == '\0') {
      *out_error = "ROI coordinate list contains an empty value";
      g_strfreev(tokens);
      return false;
    }
    char *end = nullptr;
    errno = 0;
    double value = g_ascii_strtod(stripped, &end);
    while (end != nullptr && g_ascii_isspace(*end)) {
      ++end;
    }
    if (errno != 0 || end == stripped || end == nullptr || *end != '\0' || !std::isfinite(value)) {
      *out_error = std::string("invalid ROI coordinate: ") + stripped;
      g_strfreev(tokens);
      return false;
    }
    values.push_back(value);
  }
  g_strfreev(tokens);

  if (values.size() < 6 || values.size() % 2 != 0) {
    *out_error = "ROI must contain at least three complete points";
    return false;
  }
  std::vector<Point> points;
  std::set<std::pair<long long, long long>> rounded_points;
  points.reserve(values.size() / 2);
  for (size_t i = 0; i < values.size(); i += 2) {
    if (values[i] < 0.0 || values[i] > static_cast<double>(width) ||
        values[i + 1] < 0.0 || values[i + 1] > static_cast<double>(height)) {
      *out_error = "ROI coordinate lies outside the configured frame bounds";
      return false;
    }
    points.push_back(Point{values[i], values[i + 1]});
    rounded_points.insert(
        {std::llround(values[i]), std::llround(values[i + 1])});
  }
  if (rounded_points.size() != points.size()) {
    *out_error = "ROI contains duplicate points after integer rendering";
    return false;
  }
  long double twice_area = 0.0;
  for (size_t index = 0; index < points.size(); ++index) {
    const Point &current = points[index];
    const Point &next = points[(index + 1) % points.size()];
    twice_area += static_cast<long double>(std::llround(current.x)) *
                      static_cast<long double>(std::llround(next.y)) -
                  static_cast<long double>(std::llround(next.x)) *
                      static_cast<long double>(std::llround(current.y));
  }
  if (std::fabs(twice_area) < 1.0L) {
    *out_error = "ROI is collinear after integer rendering";
    return false;
  }
  *out_points = std::move(points);
  return true;
}

static bool starts_with(const char *value, const char *prefix) {
  return value != nullptr && g_str_has_prefix(value, prefix);
}

static std::string trim_ascii(const std::string &value) {
  size_t first = 0;
  while (first < value.size() && g_ascii_isspace(value[first])) {
    ++first;
  }
  size_t last = value.size();
  while (last > first && g_ascii_isspace(value[last - 1])) {
    --last;
  }
  return value.substr(first, last - first);
}

static bool validate_unique_ini_keys(const std::string &payload,
                                     std::string *out_error) {
  std::set<std::string> sections;
  std::map<std::string, std::set<std::string>> keys_by_section;
  std::string section;
  size_t cursor = 0;
  size_t line_number = 0;
  while (cursor <= payload.size()) {
    const size_t newline = payload.find('\n', cursor);
    const size_t end = newline == std::string::npos ? payload.size() : newline;
    std::string line = payload.substr(cursor, end - cursor);
    ++line_number;
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    line = trim_ascii(line);
    if (!line.empty() && line[0] != '#' && line[0] != ';') {
      if (line.front() == '[') {
        if (line.size() < 3 || line.back() != ']') {
          *out_error = "invalid INI section syntax at line " +
                       std::to_string(line_number);
          return false;
        }
        section = trim_ascii(line.substr(1, line.size() - 2));
        if (section.empty()) {
          *out_error = "empty INI section at line " +
                       std::to_string(line_number);
          return false;
        }
        if (!sections.insert(section).second) {
          *out_error = "duplicate INI section: " + section;
          return false;
        }
      } else {
        if (section.empty()) {
          *out_error = "INI key appears before a section at line " +
                       std::to_string(line_number);
          return false;
        }
        const size_t separator = line.find('=');
        if (separator == std::string::npos) {
          *out_error = "invalid INI key syntax at line " +
                       std::to_string(line_number);
          return false;
        }
        const std::string key = trim_ascii(line.substr(0, separator));
        if (key.empty()) {
          *out_error = "empty INI key at line " + std::to_string(line_number);
          return false;
        }
        if (!keys_by_section[section].insert(key).second) {
          *out_error = "duplicate INI key " + section + "." + key;
          return false;
        }
      }
    }
    if (newline == std::string::npos) {
      break;
    }
    cursor = newline + 1;
  }
  return true;
}

static bool read_config_file(const char *config_file, std::string *out_payload,
                             std::string *out_error) {
  const int flags = O_RDONLY | O_CLOEXEC | O_NOFOLLOW;
  const int descriptor = open(config_file, flags);
  if (descriptor < 0) {
    *out_error = std::string("cannot open config-file: ") + g_strerror(errno);
    return false;
  }

  struct stat info {};
  if (fstat(descriptor, &info) != 0) {
    *out_error = std::string("cannot inspect config-file: ") + g_strerror(errno);
    close(descriptor);
    return false;
  }
  if (!S_ISREG(info.st_mode) || info.st_nlink != 1) {
    *out_error = "config-file must be a single-link regular file";
    close(descriptor);
    return false;
  }
  if (info.st_size <= 0 || static_cast<uint64_t>(info.st_size) > kMaxConfigBytes) {
    *out_error = "config-file is empty or exceeds the one MiB limit";
    close(descriptor);
    return false;
  }

  std::string payload;
  payload.reserve(static_cast<size_t>(info.st_size));
  char buffer[65536];
  while (payload.size() <= kMaxConfigBytes) {
    const size_t remaining = kMaxConfigBytes + 1 - payload.size();
    const ssize_t count = read(descriptor, buffer, std::min(sizeof(buffer), remaining));
    if (count < 0) {
      if (errno == EINTR) {
        continue;
      }
      *out_error = std::string("cannot read config-file: ") + g_strerror(errno);
      close(descriptor);
      return false;
    }
    if (count == 0) {
      break;
    }
    payload.append(buffer, static_cast<size_t>(count));
  }
  close(descriptor);
  if (payload.empty() || payload.size() > kMaxConfigBytes) {
    *out_error = "config-file is empty or exceeds the one MiB limit";
    return false;
  }
  *out_payload = std::move(payload);
  return true;
}

static bool load_config_file(const char *config_file, RuntimeConfig *out_config,
                             std::string *out_sha256,
                             std::string *out_error) {
  RuntimeConfig loaded;
  if (config_file == nullptr || *config_file == '\0') {
    *out_error = "nvdsroiexclude requires config-file";
    return false;
  }

  std::string payload;
  if (!read_config_file(config_file, &payload, out_error)) {
    return false;
  }
  if (!validate_unique_ini_keys(payload, out_error)) {
    return false;
  }
  gchar *checksum = g_compute_checksum_for_data(
      G_CHECKSUM_SHA256, reinterpret_cast<const guchar *>(payload.data()),
      payload.size());
  if (checksum == nullptr) {
    *out_error = "failed to hash config-file";
    return false;
  }

  GKeyFile *key_file = g_key_file_new();
  GError *error = nullptr;
  if (!g_key_file_load_from_data(key_file, payload.data(), payload.size(),
                                 G_KEY_FILE_NONE, &error)) {
    *out_error = std::string("failed to parse config-file: ") +
                 (error ? error->message : "unknown error");
    if (error != nullptr) {
      g_error_free(error);
    }
    g_key_file_unref(key_file);
    g_free(checksum);
    return false;
  }

  if (!g_key_file_has_group(key_file, "property")) {
    *out_error = "config-file must contain a [property] group";
    g_key_file_unref(key_file);
    g_free(checksum);
    return false;
  }
  if (!parse_bool(key_file, "property", "enable", true, &loaded.enabled,
                  out_error) ||
      !parse_int(key_file, "property", "osd-mode", 0, &loaded.osd_mode,
                 out_error) ||
      !parse_int(key_file, "property", "display-font-size", 4,
                 &loaded.display_font_size, out_error) ||
      !parse_int(key_file, "property", "config-width", 1920,
                 &loaded.config_width, out_error) ||
      !parse_int(key_file, "property", "config-height", 1080,
                 &loaded.config_height, out_error)) {
    g_key_file_unref(key_file);
    g_free(checksum);
    return false;
  }
  if (!loaded.enabled) {
    *out_error = "top-level ROI exclusion cannot be disabled while the component is active";
    g_key_file_unref(key_file);
    g_free(checksum);
    return false;
  }
  if (loaded.config_width <= 0 || loaded.config_height <= 0 ||
      loaded.config_width > kMaxConfigDimension ||
      loaded.config_height > kMaxConfigDimension) {
    *out_error = "config dimensions must be within 1..16384";
    g_key_file_unref(key_file);
    g_free(checksum);
    return false;
  }

  gsize group_count = 0;
  gchar **groups = g_key_file_get_groups(key_file, &group_count);
  for (gsize gi = 0; gi < group_count; ++gi) {
    const char *group = groups[gi];
    constexpr const char *prefix = "roi-filtering-stream-";
    if (!starts_with(group, prefix)) {
      continue;
    }

    char *end = nullptr;
    errno = 0;
    long stream_id = std::strtol(group + std::strlen(prefix), &end, 10);
    if (errno != 0 || end == group + std::strlen(prefix) || *end != '\0' ||
        stream_id < 0 || stream_id > std::numeric_limits<int>::max()) {
      *out_error = std::string("invalid stream group name: ") + group;
      g_strfreev(groups);
      g_key_file_unref(key_file);
      g_free(checksum);
      return false;
    }
    const int stream_key = static_cast<int>(stream_id);
    if (loaded.streams.find(stream_key) != loaded.streams.end()) {
      *out_error = std::string("duplicate stream group: ") + group;
      g_strfreev(groups);
      g_key_file_unref(key_file);
      g_free(checksum);
      return false;
    }

    StreamConfig stream;
    if (!parse_bool(key_file, group, "enable", true, &stream.enabled,
                    out_error) ||
        !parse_int(key_file, group, "class-id", -1, &stream.class_id,
                   out_error) ||
        !parse_bool(key_file, group, "inverse-roi", false,
                    &stream.inverse_roi, out_error) ||
        !parse_int(key_file, group, "config-width", loaded.config_width,
                   &stream.config_width, out_error) ||
        !parse_int(key_file, group, "config-height", loaded.config_height,
                   &stream.config_height, out_error)) {
      g_strfreev(groups);
      g_key_file_unref(key_file);
      g_free(checksum);
      return false;
    }
    if (stream.class_id < -1 || stream.config_width <= 0 ||
        stream.config_height <= 0 || stream.config_width > kMaxConfigDimension ||
        stream.config_height > kMaxConfigDimension) {
      *out_error = std::string("invalid stream settings in group: ") + group;
      g_strfreev(groups);
      g_key_file_unref(key_file);
      g_free(checksum);
      return false;
    }

    gsize key_count = 0;
    gchar **keys = g_key_file_get_keys(key_file, group, &key_count, nullptr);
    for (gsize ki = 0; ki < key_count; ++ki) {
      const char *key = keys[ki];
      if (!starts_with(key, "roi-")) {
        continue;
      }
      const char *label = key + 4;
      if (!valid_label(label)) {
        *out_error = std::string("invalid ROI label in group ") + group;
        g_strfreev(keys);
        g_strfreev(groups);
        g_key_file_unref(key_file);
        g_free(checksum);
        return false;
      }
      std::string enable_key = std::string("enable-") + label;
      bool roi_enabled = true;
      if (!parse_bool(key_file, group, enable_key.c_str(), true, &roi_enabled,
                      out_error)) {
        g_strfreev(keys);
        g_strfreev(groups);
        g_key_file_unref(key_file);
        g_free(checksum);
        return false;
      }
      gchar *raw = g_key_file_get_string(key_file, group, key, nullptr);
      std::vector<Point> points;
      const bool points_ok = parse_points(raw, stream.config_width,
                                          stream.config_height, &points,
                                          out_error);
      g_free(raw);
      if (!points_ok) {
        g_strfreev(keys);
        g_strfreev(groups);
        g_key_file_unref(key_file);
        g_free(checksum);
        return false;
      }
      stream.polygons.push_back(Polygon{label, std::move(points), roi_enabled});
    }
    g_strfreev(keys);

    if (stream.enabled && stream.polygons.empty()) {
      *out_error = std::string("stream group has no valid ROI polygons: ") + group;
      g_strfreev(groups);
      g_key_file_unref(key_file);
      g_free(checksum);
      return false;
    }

    loaded.streams[stream_key] = std::move(stream);
  }
  g_strfreev(groups);
  g_key_file_unref(key_file);

  if (loaded.streams.empty()) {
    *out_error = "config-file must contain at least one ROI stream group";
    g_free(checksum);
    return false;
  }

  *out_config = std::move(loaded);
  *out_sha256 = checksum;
  g_free(checksum);
  GST_INFO("validated ROI exclude config with %zu stream groups sha256=%s",
           out_config->streams.size(), out_sha256->c_str());
  return true;
}

static bool point_on_segment(const Point &p, const Point &a, const Point &b) {
  constexpr double eps = 1e-6;
  const double cross = (p.y - a.y) * (b.x - a.x) - (p.x - a.x) * (b.y - a.y);
  if (std::fabs(cross) > eps) {
    return false;
  }
  return p.x >= std::min(a.x, b.x) - eps && p.x <= std::max(a.x, b.x) + eps &&
         p.y >= std::min(a.y, b.y) - eps && p.y <= std::max(a.y, b.y) + eps;
}

static bool point_in_polygon(const Point &p, const std::vector<Point> &poly) {
  if (poly.size() < 3) {
    return false;
  }
  bool inside = false;
  for (size_t i = 0, j = poly.size() - 1; i < poly.size(); j = i++) {
    const Point &a = poly[i];
    const Point &b = poly[j];
    if (point_on_segment(p, a, b)) {
      return true;
    }
    const bool intersects = ((a.y > p.y) != (b.y > p.y)) &&
                            (p.x < (b.x - a.x) * (p.y - a.y) / ((b.y - a.y) + 1e-12) + a.x);
    if (intersects) {
      inside = !inside;
    }
  }
  return inside;
}

static MappedPolygon map_polygon(const Polygon &poly, const StreamConfig &stream, const NvDsFrameMeta *frame_meta) {
  const int cfg_w = stream.config_width > 0 ? stream.config_width : static_cast<int>(frame_meta->pipeline_width);
  const int cfg_h = stream.config_height > 0 ? stream.config_height : static_cast<int>(frame_meta->pipeline_height);
  const double out_w = frame_meta->pipeline_width > 0 ? frame_meta->pipeline_width : cfg_w;
  const double out_h = frame_meta->pipeline_height > 0 ? frame_meta->pipeline_height : cfg_h;
  const double scale = std::min(out_w / std::max(1, cfg_w), out_h / std::max(1, cfg_h));
  const double dx = (out_w - cfg_w * scale) * 0.5;
  const double dy = (out_h - cfg_h * scale) * 0.5;

  MappedPolygon mapped;
  mapped.label = poly.label;
  mapped.points.reserve(poly.points.size());
  for (const Point &p : poly.points) {
    mapped.points.push_back(Point{p.x * scale + dx, p.y * scale + dy});
  }
  return mapped;
}

static bool bbox_fully_inside_any_roi(const NvDsObjectMeta *obj_meta, const std::vector<MappedPolygon> &polygons) {
  const double left = obj_meta->rect_params.left;
  const double top = obj_meta->rect_params.top;
  const double right = left + obj_meta->rect_params.width;
  const double bottom = top + obj_meta->rect_params.height;
  const Point corners[] = {{left, top}, {right, top}, {right, bottom}, {left, bottom}};

  for (const MappedPolygon &poly : polygons) {
    bool all_inside = true;
    for (const Point &corner : corners) {
      if (!point_in_polygon(corner, poly.points)) {
        all_inside = false;
        break;
      }
    }
    if (all_inside) {
      return true;
    }
  }
  return false;
}

static void add_roi_display_meta(NvDsBatchMeta *batch_meta, NvDsFrameMeta *frame_meta,
                                 const std::vector<MappedPolygon> &polygons) {
  NvDsDisplayMeta *display_meta = nullptr;

  auto ensure_display_meta = [&]() -> NvDsDisplayMeta * {
    if (display_meta == nullptr || display_meta->num_lines >= MAX_ELEMENTS_IN_DISPLAY_META) {
      if (display_meta != nullptr) {
        nvds_add_display_meta_to_frame(frame_meta, display_meta);
      }
      display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
      display_meta->num_lines = 0;
    }
    return display_meta;
  };

  for (const MappedPolygon &poly : polygons) {
    if (poly.points.size() < 2) {
      continue;
    }
    for (size_t i = 0; i < poly.points.size(); ++i) {
      NvDsDisplayMeta *meta = ensure_display_meta();
      NvOSD_LineParams &line = meta->line_params[meta->num_lines++];
      const Point &a = poly.points[i];
      const Point &b = poly.points[(i + 1) % poly.points.size()];
      line.x1 = static_cast<unsigned int>(std::max(0.0, std::round(a.x)));
      line.y1 = static_cast<unsigned int>(std::max(0.0, std::round(a.y)));
      line.x2 = static_cast<unsigned int>(std::max(0.0, std::round(b.x)));
      line.y2 = static_cast<unsigned int>(std::max(0.0, std::round(b.y)));
      line.line_width = 3;
      line.line_color = NvOSD_ColorParams{0.0, 1.0, 0.0, 1.0};
    }
  }

  if (display_meta != nullptr) {
    nvds_add_display_meta_to_frame(frame_meta, display_meta);
  }
}

} // namespace

typedef struct _GstNvDsROIExclude {
  GstBaseTransform parent;
  GMutex lock;
  gchar *config_file;
  gchar *id_mode;
  RuntimeConfig *config;
  guint reload_request_sequence;
  guint reload_accepted_sequence;
  guint reload_failed_sequence;
  guint reload_error_count;
  guint objects_removed_count;
  gboolean last_reload_ok;
  gchar *expected_config_sha256;
  gchar *active_config_sha256;
  gchar *last_reload_error;
} GstNvDsROIExclude;

typedef struct _GstNvDsROIExcludeClass {
  GstBaseTransformClass parent_class;
} GstNvDsROIExcludeClass;

G_DEFINE_TYPE(GstNvDsROIExclude, gst_nvds_roi_exclude, GST_TYPE_BASE_TRANSFORM)

static std::string snapshot_config_file(GstNvDsROIExclude *self) {
  g_mutex_lock(&self->lock);
  const std::string path = self->config_file != nullptr ? self->config_file : "";
  g_mutex_unlock(&self->lock);
  return path;
}

static std::string snapshot_id_mode(GstNvDsROIExclude *self) {
  g_mutex_lock(&self->lock);
  const std::string mode = self->id_mode != nullptr ? self->id_mode : "";
  g_mutex_unlock(&self->lock);
  return mode;
}

static bool valid_sha256(const std::string &value) {
  if (value.size() != 64) {
    return false;
  }
  return std::all_of(value.begin(), value.end(), [](unsigned char ch) {
    return (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f');
  });
}

static void record_reload_failure(GstNvDsROIExclude *self, guint sequence,
                                  const std::string &error) {
  gboolean ok_changed = FALSE;
  g_mutex_lock(&self->lock);
  const guint previous_failed = self->reload_failed_sequence;
  ok_changed = self->last_reload_ok != FALSE;
  self->reload_failed_sequence = sequence;
  if (self->reload_error_count < G_MAXUINT) {
    ++self->reload_error_count;
  }
  self->last_reload_ok = FALSE;
  g_free(self->last_reload_error);
  self->last_reload_error = g_strdup(error.c_str());
  g_mutex_unlock(&self->lock);

  if (previous_failed != sequence) {
    g_object_notify_by_pspec(G_OBJECT(self),
                             properties[kPropertyReloadFailedSequence]);
  }
  if (ok_changed) {
    g_object_notify_by_pspec(G_OBJECT(self), properties[kPropertyLastReloadOk]);
  }
  g_object_notify_by_pspec(G_OBJECT(self), properties[kPropertyReloadErrorCount]);
  g_object_notify_by_pspec(G_OBJECT(self), properties[kPropertyLastReloadError]);
}

static bool load_requested_config(GstNvDsROIExclude *self, guint sequence,
                                  bool require_expected_sha256,
                                  std::string *out_error) {
  const std::string config_file = snapshot_config_file(self);
  std::string expected_sha256;
  if (require_expected_sha256) {
    g_mutex_lock(&self->lock);
    expected_sha256 = self->expected_config_sha256 != nullptr
                          ? self->expected_config_sha256
                          : "";
    g_mutex_unlock(&self->lock);
    if (!valid_sha256(expected_sha256)) {
      *out_error = "expected-config-sha256 must be a lowercase SHA-256 digest";
      record_reload_failure(self, sequence, *out_error);
      return false;
    }
  }
  RuntimeConfig loaded;
  std::string sha256;
  if (!load_config_file(config_file.c_str(), &loaded, &sha256, out_error)) {
    record_reload_failure(self, sequence, *out_error);
    return false;
  }
  if (require_expected_sha256 && sha256 != expected_sha256) {
    *out_error = "config-file SHA-256 does not match expected-config-sha256";
    record_reload_failure(self, sequence, *out_error);
    return false;
  }

  gboolean ok_changed = FALSE;
  guint previous_accepted = 0;
  g_mutex_lock(&self->lock);
  previous_accepted = self->reload_accepted_sequence;
  ok_changed = self->last_reload_ok == FALSE;
  *(self->config) = std::move(loaded);
  self->reload_accepted_sequence = sequence;
  self->last_reload_ok = TRUE;
  g_free(self->active_config_sha256);
  self->active_config_sha256 = g_strdup(sha256.c_str());
  g_free(self->last_reload_error);
  self->last_reload_error = g_strdup("");
  g_mutex_unlock(&self->lock);

  if (previous_accepted != sequence) {
    g_object_notify_by_pspec(G_OBJECT(self),
                             properties[kPropertyReloadAcceptedSequence]);
  }
  if (ok_changed) {
    g_object_notify_by_pspec(G_OBJECT(self), properties[kPropertyLastReloadOk]);
  }
  g_object_notify_by_pspec(G_OBJECT(self),
                           properties[kPropertyActiveConfigSha256]);
  g_object_notify_by_pspec(G_OBJECT(self), properties[kPropertyLastReloadError]);
  return true;
}

static void gst_nvds_roi_exclude_set_property(GObject *object, guint prop_id, const GValue *value, GParamSpec *pspec) {
  GstNvDsROIExclude *self = reinterpret_cast<GstNvDsROIExclude *>(object);
  switch (prop_id) {
  case kPropertyConfigFile:
    g_mutex_lock(&self->lock);
    g_free(self->config_file);
    self->config_file = g_value_dup_string(value);
    g_mutex_unlock(&self->lock);
    break;
  case kPropertyIdMode:
    g_mutex_lock(&self->lock);
    g_free(self->id_mode);
    self->id_mode = g_value_dup_string(value);
    g_mutex_unlock(&self->lock);
    break;
  case kPropertyReloadRequestSequence: {
    const guint requested = g_value_get_uint(value);
    g_mutex_lock(&self->lock);
    if (requested <= self->reload_request_sequence) {
      GST_WARNING_OBJECT(
          self,
          "ignoring non-monotonic ROI reload request sequence=%u current=%u",
          requested, self->reload_request_sequence);
      g_mutex_unlock(&self->lock);
      return;
    }
    self->reload_request_sequence = requested;
    self->last_reload_ok = FALSE;
    g_free(self->last_reload_error);
    self->last_reload_error = g_strdup("reload_pending");
    g_mutex_unlock(&self->lock);
    g_object_notify_by_pspec(object,
                             properties[kPropertyReloadRequestSequence]);
    g_object_notify_by_pspec(object, properties[kPropertyLastReloadOk]);
    g_object_notify_by_pspec(object, properties[kPropertyLastReloadError]);
    std::string error;
    if (!load_requested_config(self, requested, true, &error)) {
      GST_ERROR_OBJECT(self,
                       "rejected ROI exclusion reload request sequence=%u: %s",
                       requested, error.c_str());
    } else {
      GST_INFO_OBJECT(self,
                      "accepted ROI exclusion reload request sequence=%u",
                      requested);
    }
    break;
  }
  case kPropertyExpectedConfigSha256: {
    const gchar *expected = g_value_get_string(value);
    g_mutex_lock(&self->lock);
    g_free(self->expected_config_sha256);
    self->expected_config_sha256 = g_strdup(expected != nullptr ? expected : "");
    g_mutex_unlock(&self->lock);
    break;
  }
  default:
    G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
    break;
  }
}

static void gst_nvds_roi_exclude_get_property(GObject *object, guint prop_id, GValue *value, GParamSpec *pspec) {
  GstNvDsROIExclude *self = reinterpret_cast<GstNvDsROIExclude *>(object);
  g_mutex_lock(&self->lock);
  switch (prop_id) {
  case kPropertyConfigFile:
    g_value_set_string(value, self->config_file);
    break;
  case kPropertyIdMode:
    g_value_set_string(value, self->id_mode);
    break;
  case kPropertyReloadRequestSequence:
    g_value_set_uint(value, self->reload_request_sequence);
    break;
  case kPropertyReloadAcceptedSequence:
    g_value_set_uint(value, self->reload_accepted_sequence);
    break;
  case kPropertyReloadFailedSequence:
    g_value_set_uint(value, self->reload_failed_sequence);
    break;
  case kPropertyLastReloadOk:
    g_value_set_boolean(value, self->last_reload_ok);
    break;
  case kPropertyExpectedConfigSha256:
    g_value_set_string(value, self->expected_config_sha256);
    break;
  case kPropertyActiveConfigSha256:
    g_value_set_string(value, self->active_config_sha256);
    break;
  case kPropertyReloadErrorCount:
    g_value_set_uint(value, self->reload_error_count);
    break;
  case kPropertyObjectsRemovedCount:
    g_value_set_uint(value, self->objects_removed_count);
    break;
  case kPropertyLastReloadError:
    g_value_set_string(value, self->last_reload_error);
    break;
  default:
    g_mutex_unlock(&self->lock);
    G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
    return;
  }
  g_mutex_unlock(&self->lock);
}

static void gst_nvds_roi_exclude_finalize(GObject *object) {
  GstNvDsROIExclude *self = reinterpret_cast<GstNvDsROIExclude *>(object);
  g_free(self->config_file);
  g_free(self->id_mode);
  g_free(self->expected_config_sha256);
  g_free(self->active_config_sha256);
  g_free(self->last_reload_error);
  delete self->config;
  g_mutex_clear(&self->lock);
  G_OBJECT_CLASS(gst_nvds_roi_exclude_parent_class)->finalize(object);
}

static gboolean gst_nvds_roi_exclude_start(GstBaseTransform *trans) {
  GstNvDsROIExclude *self = reinterpret_cast<GstNvDsROIExclude *>(trans);
  const std::string id_mode = snapshot_id_mode(self);
  if (id_mode != "source-id" && id_mode != "pad-index") {
    const std::string error = "id-mode must be source-id or pad-index";
    record_reload_failure(self, 0, error);
    GST_ERROR_OBJECT(self, "%s", error.c_str());
    return FALSE;
  }
  std::string error;
  if (!load_requested_config(self, 0, false, &error)) {
    GST_ERROR_OBJECT(self, "initial ROI exclusion config load failed: %s",
                     error.c_str());
    return FALSE;
  }
  return TRUE;
}

static GstFlowReturn gst_nvds_roi_exclude_transform_ip(GstBaseTransform *trans, GstBuffer *buf) {
  GstNvDsROIExclude *self = reinterpret_cast<GstNvDsROIExclude *>(trans);
  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);
  if (batch_meta == nullptr) {
    return GST_FLOW_OK;
  }

  g_mutex_lock(&self->lock);
  if (self->config == nullptr || !self->config->enabled) {
    g_mutex_unlock(&self->lock);
    GST_ELEMENT_ERROR(self, RESOURCE, SETTINGS,
                      ("ROI exclusion has no active validated config"),
                      ("component cannot safely process metadata"));
    return GST_FLOW_ERROR;
  }
  const bool use_pad_index = g_strcmp0(self->id_mode, "pad-index") == 0;
  nvds_acquire_meta_lock(batch_meta);
  for (NvDsMetaList *frame_list = batch_meta->frame_meta_list; frame_list != nullptr; frame_list = frame_list->next) {
    NvDsFrameMeta *frame_meta = reinterpret_cast<NvDsFrameMeta *>(frame_list->data);
    if (frame_meta == nullptr) {
      continue;
    }

    const int stream_key = use_pad_index ? static_cast<int>(frame_meta->pad_index) : static_cast<int>(frame_meta->source_id);
    auto stream_it = self->config->streams.find(stream_key);
    if (stream_it == self->config->streams.end()) {
      nvds_release_meta_lock(batch_meta);
      g_mutex_unlock(&self->lock);
      GST_ELEMENT_ERROR(self, RESOURCE, SETTINGS,
                        ("ROI exclusion config is missing source %d", stream_key),
                        ("configured analytics stream coverage drifted at runtime"));
      return GST_FLOW_ERROR;
    }
    const StreamConfig &stream = stream_it->second;
    if (!stream.enabled) {
      continue;
    }

    std::vector<MappedPolygon> mapped;
    for (const Polygon &poly : stream.polygons) {
      if (poly.enabled) {
        mapped.push_back(map_polygon(poly, stream, frame_meta));
      }
    }
    if (mapped.empty()) {
      continue;
    }

    NvDsMetaList *obj_list = frame_meta->obj_meta_list;
    while (obj_list != nullptr) {
      NvDsObjectMeta *obj_meta = reinterpret_cast<NvDsObjectMeta *>(obj_list->data);
      NvDsMetaList *next = obj_list->next;
      if (obj_meta != nullptr && (stream.class_id < 0 || obj_meta->class_id == stream.class_id)) {
        const bool inside = bbox_fully_inside_any_roi(obj_meta, mapped);
        const bool remove = stream.inverse_roi ? !inside : inside;
        if (remove) {
          nvds_remove_obj_meta_from_frame(frame_meta, obj_meta);
          if (self->objects_removed_count < G_MAXUINT) {
            ++self->objects_removed_count;
          }
        }
      }
      obj_list = next;
    }

    if (self->config->osd_mode != 0) {
      add_roi_display_meta(batch_meta, frame_meta, mapped);
    }
  }
  nvds_release_meta_lock(batch_meta);
  g_mutex_unlock(&self->lock);
  return GST_FLOW_OK;
}

static void gst_nvds_roi_exclude_init(GstNvDsROIExclude *self) {
  g_mutex_init(&self->lock);
  self->config_file = nullptr;
  self->id_mode = g_strdup(g_getenv("NOESIS_DS_EXCLUDE_ID_MODE") ? g_getenv("NOESIS_DS_EXCLUDE_ID_MODE") : "source-id");
  self->config = new RuntimeConfig();
  self->reload_request_sequence = 0;
  self->reload_accepted_sequence = 0;
  self->reload_failed_sequence = 0;
  self->reload_error_count = 0;
  self->objects_removed_count = 0;
  self->last_reload_ok = FALSE;
  self->expected_config_sha256 = g_strdup("");
  self->active_config_sha256 = g_strdup("");
  self->last_reload_error = g_strdup("not_loaded");
  gst_base_transform_set_in_place(GST_BASE_TRANSFORM(self), TRUE);
  gst_base_transform_set_passthrough(GST_BASE_TRANSFORM(self), FALSE);
}

static void gst_nvds_roi_exclude_class_init(GstNvDsROIExcludeClass *klass) {
  GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
  GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
  GstBaseTransformClass *base_transform_class = GST_BASE_TRANSFORM_CLASS(klass);

  gobject_class->set_property = gst_nvds_roi_exclude_set_property;
  gobject_class->get_property = gst_nvds_roi_exclude_get_property;
  gobject_class->finalize = gst_nvds_roi_exclude_finalize;

  properties[kPropertyConfigFile] = g_param_spec_string(
      "config-file", "Config file", "Path to nvdsroiexclude INI config",
      nullptr,
      static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
                               GST_PARAM_MUTABLE_READY));
  properties[kPropertyIdMode] = g_param_spec_string(
      "id-mode", "ID mode", "Stream key mode: source-id or pad-index",
      "source-id",
      static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
                               GST_PARAM_MUTABLE_READY));
  properties[kPropertyReloadRequestSequence] = g_param_spec_uint(
      "reload-request-sequence", "Reload request sequence",
      "Strictly monotonic sequence requesting one validated config reload", 0,
      G_MAXUINT, 0,
      static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
                               G_PARAM_EXPLICIT_NOTIFY |
                               GST_PARAM_MUTABLE_PLAYING));
  properties[kPropertyReloadAcceptedSequence] = g_param_spec_uint(
      "reload-accepted-sequence", "Reload accepted sequence",
      "Most recent reload sequence parsed and activated on the streaming path",
      0, G_MAXUINT, 0,
      static_cast<GParamFlags>(G_PARAM_READABLE | G_PARAM_STATIC_STRINGS));
  properties[kPropertyReloadFailedSequence] = g_param_spec_uint(
      "reload-failed-sequence", "Reload failed sequence",
      "Most recent reload sequence rejected by strict config validation", 0,
      G_MAXUINT, 0,
      static_cast<GParamFlags>(G_PARAM_READABLE | G_PARAM_STATIC_STRINGS));
  properties[kPropertyLastReloadOk] = g_param_spec_boolean(
      "last-reload-ok", "Last reload succeeded",
      "Whether the initial load or most recent request was applied", FALSE,
      static_cast<GParamFlags>(G_PARAM_READABLE | G_PARAM_STATIC_STRINGS));
  properties[kPropertyExpectedConfigSha256] = g_param_spec_string(
      "expected-config-sha256", "Expected config SHA-256",
      "Required SHA-256 for the next monotonic reload request", "",
      static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
                               GST_PARAM_MUTABLE_PLAYING));
  properties[kPropertyActiveConfigSha256] = g_param_spec_string(
      "active-config-sha256", "Active config SHA-256",
      "SHA-256 of the exact INI bytes parsed into the active config", "",
      static_cast<GParamFlags>(G_PARAM_READABLE | G_PARAM_STATIC_STRINGS));
  properties[kPropertyReloadErrorCount] = g_param_spec_uint(
      "reload-error-count", "Reload error count",
      "Number of rejected initial loads or monotonic reload requests", 0,
      G_MAXUINT, 0,
      static_cast<GParamFlags>(G_PARAM_READABLE | G_PARAM_STATIC_STRINGS));
  properties[kPropertyObjectsRemovedCount] = g_param_spec_uint(
      "objects-removed-count", "Objects removed count",
      "Number of object metadata records removed by the active ROI config", 0,
      G_MAXUINT, 0,
      static_cast<GParamFlags>(G_PARAM_READABLE | G_PARAM_STATIC_STRINGS));
  properties[kPropertyLastReloadError] = g_param_spec_string(
      "last-reload-error", "Last reload error",
      "Empty after success; diagnostic reason after pending or failed reload",
      "not_loaded",
      static_cast<GParamFlags>(G_PARAM_READABLE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_properties(gobject_class, kPropertyCount, properties);

  gst_element_class_set_static_metadata(element_class, "Noesis ROI Exclude", "Filter/Metadata",
                                        "Removes NvDs object metadata fully inside configured static ROIs",
                                        "Noesis");

  GstCaps *caps = gst_caps_new_any();
  gst_element_class_add_pad_template(element_class,
                                     gst_pad_template_new("sink", GST_PAD_SINK, GST_PAD_ALWAYS, gst_caps_ref(caps)));
  gst_element_class_add_pad_template(element_class,
                                     gst_pad_template_new("src", GST_PAD_SRC, GST_PAD_ALWAYS, caps));

  base_transform_class->start = GST_DEBUG_FUNCPTR(gst_nvds_roi_exclude_start);
  base_transform_class->transform_ip = GST_DEBUG_FUNCPTR(gst_nvds_roi_exclude_transform_ip);
}

static gboolean plugin_init(GstPlugin *plugin) {
  GST_DEBUG_CATEGORY_INIT(gst_nvds_roi_exclude_debug, "nvdsroiexclude", 0, "Noesis ROI exclusion");
  return gst_element_register(plugin, "nvdsroiexclude", GST_RANK_NONE, gst_nvds_roi_exclude_get_type());
}

GST_PLUGIN_DEFINE(GST_VERSION_MAJOR, GST_VERSION_MINOR, nvdsroiexclude, "Noesis ROI exclusion plugin", plugin_init,
                  "1.0", "Proprietary", "Noesis", "https://github.com/noesis")
