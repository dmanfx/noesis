#include <gst/base/gstbasetransform.h>
#include <gst/gst.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <sys/stat.h>
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

static bool parse_bool(GKeyFile *key_file, const char *group, const char *key, bool fallback) {
  GError *error = nullptr;
  gboolean value = g_key_file_get_boolean(key_file, group, key, &error);
  if (error != nullptr) {
    g_error_free(error);
    return fallback;
  }
  return value != FALSE;
}

static int parse_int(GKeyFile *key_file, const char *group, const char *key, int fallback) {
  GError *error = nullptr;
  int value = g_key_file_get_integer(key_file, group, key, &error);
  if (error != nullptr) {
    g_error_free(error);
    return fallback;
  }
  return value;
}

static std::vector<Point> parse_points(const char *raw) {
  std::vector<Point> points;
  if (raw == nullptr) {
    return points;
  }

  std::vector<double> values;
  gchar **tokens = g_strsplit(raw, ";", -1);
  for (gchar **token = tokens; token != nullptr && *token != nullptr; ++token) {
    gchar *stripped = g_strstrip(*token);
    if (stripped == nullptr || *stripped == '\0') {
      continue;
    }
    char *end = nullptr;
    errno = 0;
    double value = g_ascii_strtod(stripped, &end);
    if (errno == 0 && end != stripped) {
      values.push_back(value);
    }
  }
  g_strfreev(tokens);

  if (values.size() < 6 || values.size() % 2 != 0) {
    return points;
  }
  points.reserve(values.size() / 2);
  for (size_t i = 0; i < values.size(); i += 2) {
    points.push_back(Point{values[i], values[i + 1]});
  }
  return points;
}

static bool starts_with(const char *value, const char *prefix) {
  return value != nullptr && g_str_has_prefix(value, prefix);
}

static bool load_config_file(const char *config_file, RuntimeConfig *out_config) {
  RuntimeConfig loaded;
  if (config_file == nullptr || *config_file == '\0') {
    GST_ERROR("nvdsroiexclude requires config-file");
    return false;
  }

  GKeyFile *key_file = g_key_file_new();
  GError *error = nullptr;
  if (!g_key_file_load_from_file(key_file, config_file, G_KEY_FILE_NONE, &error)) {
    GST_ERROR("failed to load config-file %s: %s", config_file, error ? error->message : "unknown error");
    if (error != nullptr) {
      g_error_free(error);
    }
    g_key_file_unref(key_file);
    return false;
  }

  if (g_key_file_has_group(key_file, "property")) {
    loaded.enabled = parse_bool(key_file, "property", "enable", true);
    loaded.osd_mode = parse_int(key_file, "property", "osd-mode", 0);
    loaded.display_font_size = parse_int(key_file, "property", "display-font-size", 4);
    loaded.config_width = parse_int(key_file, "property", "config-width", 1920);
    loaded.config_height = parse_int(key_file, "property", "config-height", 1080);
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
    long stream_id = std::strtol(group + std::strlen(prefix), &end, 10);
    if (end == group + std::strlen(prefix) || stream_id < 0) {
      continue;
    }

    StreamConfig stream;
    stream.enabled = parse_bool(key_file, group, "enable", true);
    stream.class_id = parse_int(key_file, group, "class-id", -1);
    stream.inverse_roi = parse_bool(key_file, group, "inverse-roi", false);
    stream.config_width = parse_int(key_file, group, "config-width", loaded.config_width);
    stream.config_height = parse_int(key_file, group, "config-height", loaded.config_height);

    gsize key_count = 0;
    gchar **keys = g_key_file_get_keys(key_file, group, &key_count, nullptr);
    for (gsize ki = 0; ki < key_count; ++ki) {
      const char *key = keys[ki];
      if (!starts_with(key, "roi-")) {
        continue;
      }
      const char *label = key + 4;
      std::string enable_key = std::string("enable-") + label;
      bool roi_enabled = parse_bool(key_file, group, enable_key.c_str(), true);
      gchar *raw = g_key_file_get_string(key_file, group, key, nullptr);
      std::vector<Point> points = parse_points(raw);
      g_free(raw);
      if (points.size() >= 3) {
        stream.polygons.push_back(Polygon{label, std::move(points), roi_enabled});
      }
    }
    g_strfreev(keys);

    loaded.streams[static_cast<int>(stream_id)] = std::move(stream);
  }
  g_strfreev(groups);
  g_key_file_unref(key_file);

  *out_config = std::move(loaded);
  GST_INFO("loaded ROI exclude config %s with %zu stream groups", config_file, out_config->streams.size());
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
  gchar *config_file;
  gchar *id_mode;
  RuntimeConfig *config;
  time_t config_mtime;
} GstNvDsROIExclude;

typedef struct _GstNvDsROIExcludeClass {
  GstBaseTransformClass parent_class;
} GstNvDsROIExcludeClass;

G_DEFINE_TYPE(GstNvDsROIExclude, gst_nvds_roi_exclude, GST_TYPE_BASE_TRANSFORM)

enum {
  PROP_0,
  PROP_CONFIG_FILE,
  PROP_ID_MODE,
};

static bool reload_if_needed(GstNvDsROIExclude *self, bool force) {
  if (self->config_file == nullptr || *self->config_file == '\0') {
    return false;
  }

  struct stat st {};
  if (stat(self->config_file, &st) != 0) {
    GST_ERROR_OBJECT(self, "failed to stat config-file %s: %s", self->config_file, g_strerror(errno));
    return false;
  }

  if (!force && self->config != nullptr && self->config_mtime == st.st_mtime) {
    return true;
  }

  RuntimeConfig loaded;
  if (!load_config_file(self->config_file, &loaded)) {
    return false;
  }
  *(self->config) = std::move(loaded);
  self->config_mtime = st.st_mtime;
  return true;
}

static void gst_nvds_roi_exclude_set_property(GObject *object, guint prop_id, const GValue *value, GParamSpec *pspec) {
  GstNvDsROIExclude *self = reinterpret_cast<GstNvDsROIExclude *>(object);
  switch (prop_id) {
  case PROP_CONFIG_FILE:
    g_free(self->config_file);
    self->config_file = g_value_dup_string(value);
    self->config_mtime = 0;
    break;
  case PROP_ID_MODE:
    g_free(self->id_mode);
    self->id_mode = g_value_dup_string(value);
    break;
  default:
    G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
    break;
  }
}

static void gst_nvds_roi_exclude_get_property(GObject *object, guint prop_id, GValue *value, GParamSpec *pspec) {
  GstNvDsROIExclude *self = reinterpret_cast<GstNvDsROIExclude *>(object);
  switch (prop_id) {
  case PROP_CONFIG_FILE:
    g_value_set_string(value, self->config_file);
    break;
  case PROP_ID_MODE:
    g_value_set_string(value, self->id_mode);
    break;
  default:
    G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
    break;
  }
}

static void gst_nvds_roi_exclude_finalize(GObject *object) {
  GstNvDsROIExclude *self = reinterpret_cast<GstNvDsROIExclude *>(object);
  g_free(self->config_file);
  g_free(self->id_mode);
  delete self->config;
  G_OBJECT_CLASS(gst_nvds_roi_exclude_parent_class)->finalize(object);
}

static gboolean gst_nvds_roi_exclude_start(GstBaseTransform *trans) {
  GstNvDsROIExclude *self = reinterpret_cast<GstNvDsROIExclude *>(trans);
  return reload_if_needed(self, true) ? TRUE : FALSE;
}

static GstFlowReturn gst_nvds_roi_exclude_transform_ip(GstBaseTransform *trans, GstBuffer *buf) {
  GstNvDsROIExclude *self = reinterpret_cast<GstNvDsROIExclude *>(trans);
  if (!reload_if_needed(self, false) || self->config == nullptr || !self->config->enabled) {
    return GST_FLOW_OK;
  }

  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);
  if (batch_meta == nullptr) {
    return GST_FLOW_OK;
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
      continue;
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
        }
      }
      obj_list = next;
    }

    if (self->config->osd_mode != 0) {
      add_roi_display_meta(batch_meta, frame_meta, mapped);
    }
  }
  nvds_release_meta_lock(batch_meta);
  return GST_FLOW_OK;
}

static void gst_nvds_roi_exclude_init(GstNvDsROIExclude *self) {
  self->config_file = nullptr;
  self->id_mode = g_strdup(g_getenv("NOESIS_DS_EXCLUDE_ID_MODE") ? g_getenv("NOESIS_DS_EXCLUDE_ID_MODE") : "source-id");
  self->config = new RuntimeConfig();
  self->config_mtime = 0;
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

  g_object_class_install_property(
      gobject_class, PROP_CONFIG_FILE,
      g_param_spec_string("config-file", "Config file", "Path to nvdsroiexclude INI config", nullptr,
                          static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property(
      gobject_class, PROP_ID_MODE,
      g_param_spec_string("id-mode", "ID mode", "Stream key mode: source-id or pad-index", "source-id",
                          static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

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
