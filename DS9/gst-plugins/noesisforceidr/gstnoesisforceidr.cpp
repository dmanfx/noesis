// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Noesis contributors

#include <gst/base/gstbasetransform.h>
#include <gst/gst.h>

#include <gst-nvcustomevent.h>

#ifndef PACKAGE
#define PACKAGE "noesis"
#endif

#ifndef PACKAGE_VERSION
#define PACKAGE_VERSION "1.0.0"
#endif

GST_DEBUG_CATEGORY_STATIC(gst_noesis_force_idr_debug);
#define GST_CAT_DEFAULT gst_noesis_force_idr_debug

namespace {

constexpr const char *kDefaultStreamId = "mosaic";

enum PropertyId : guint {
  kPropertyNone = 0,
  kPropertyRequestSequence,
  kPropertyAcceptedSequence,
  kPropertyLastRequestOk,
  kPropertyStreamId,
  kPropertyCount,
};

GParamSpec *properties[kPropertyCount] = {};

}  // namespace

typedef struct _GstNoesisForceIdr {
  GstBaseTransform parent;

  GMutex lock;
  guint request_sequence;
  guint accepted_sequence;
  gboolean last_request_ok;
  gchar *stream_id;
} GstNoesisForceIdr;

typedef struct _GstNoesisForceIdrClass {
  GstBaseTransformClass parent_class;
} GstNoesisForceIdrClass;

G_DEFINE_TYPE(GstNoesisForceIdr, gst_noesis_force_idr,
              GST_TYPE_BASE_TRANSFORM)

static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE(
    "sink", GST_PAD_SINK, GST_PAD_ALWAYS, GST_STATIC_CAPS_ANY);

static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE(
    "src", GST_PAD_SRC, GST_PAD_ALWAYS, GST_STATIC_CAPS_ANY);

static void gst_noesis_force_idr_set_property(GObject *object, guint property_id,
                                              const GValue *value,
                                              GParamSpec *param_spec) {
  auto *self = reinterpret_cast<GstNoesisForceIdr *>(object);

  switch (property_id) {
    case kPropertyRequestSequence: {
      const guint requested_sequence = g_value_get_uint(value);
      gboolean request_ok = FALSE;
      guint previous_accepted_sequence = 0;
      gboolean previous_request_ok = FALSE;

      g_mutex_lock(&self->lock);
      if (requested_sequence <= self->request_sequence) {
        GST_WARNING_OBJECT(
            self,
            "ignoring non-monotonic force-IDR request sequence %u "
            "(current=%u)",
            requested_sequence, self->request_sequence);
        g_mutex_unlock(&self->lock);
        return;
      }

      self->request_sequence = requested_sequence;
      previous_accepted_sequence = self->accepted_sequence;
      previous_request_ok = self->last_request_ok;

      GstEvent *event = gst_nvevent_enc_force_idr(self->stream_id, 1);
      if (event != nullptr) {
        // gst_pad_push_event() consumes the event regardless of the result.
        request_ok = gst_pad_push_event(GST_BASE_TRANSFORM_SRC_PAD(self), event);
      }

      self->last_request_ok = request_ok;
      if (request_ok) {
        self->accepted_sequence = requested_sequence;
        GST_INFO_OBJECT(
            self,
            "accepted force-IDR request sequence=%u stream-id=%s",
            requested_sequence, self->stream_id);
      } else {
        GST_ERROR_OBJECT(
            self,
            "downstream rejected force-IDR request sequence=%u stream-id=%s",
            requested_sequence, self->stream_id);
      }
      g_mutex_unlock(&self->lock);

      g_object_notify_by_pspec(object, properties[kPropertyRequestSequence]);
      if (request_ok && previous_accepted_sequence != requested_sequence) {
        g_object_notify_by_pspec(object, properties[kPropertyAcceptedSequence]);
      }
      if (previous_request_ok != request_ok) {
        g_object_notify_by_pspec(object, properties[kPropertyLastRequestOk]);
      }
      return;
    }
    case kPropertyStreamId: {
      const gchar *stream_id = g_value_get_string(value);
      if (stream_id == nullptr || *stream_id == '\0') {
        GST_WARNING_OBJECT(self, "ignoring empty stream-id");
        return;
      }
      g_mutex_lock(&self->lock);
      if (g_strcmp0(self->stream_id, stream_id) != 0) {
        g_free(self->stream_id);
        self->stream_id = g_strdup(stream_id);
      }
      g_mutex_unlock(&self->lock);
      return;
    }
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID(object, property_id, param_spec);
      return;
  }
}

static void gst_noesis_force_idr_get_property(GObject *object, guint property_id,
                                              GValue *value,
                                              GParamSpec *param_spec) {
  auto *self = reinterpret_cast<GstNoesisForceIdr *>(object);

  g_mutex_lock(&self->lock);
  switch (property_id) {
    case kPropertyRequestSequence:
      g_value_set_uint(value, self->request_sequence);
      break;
    case kPropertyAcceptedSequence:
      g_value_set_uint(value, self->accepted_sequence);
      break;
    case kPropertyLastRequestOk:
      g_value_set_boolean(value, self->last_request_ok);
      break;
    case kPropertyStreamId:
      g_value_set_string(value, self->stream_id);
      break;
    default:
      g_mutex_unlock(&self->lock);
      G_OBJECT_WARN_INVALID_PROPERTY_ID(object, property_id, param_spec);
      return;
  }
  g_mutex_unlock(&self->lock);
}

static GstFlowReturn gst_noesis_force_idr_transform_ip(GstBaseTransform *,
                                                       GstBuffer *) {
  // Metadata and NVMM surfaces remain untouched; this element only emits a
  // downstream control event when request-sequence advances.
  return GST_FLOW_OK;
}

static void gst_noesis_force_idr_finalize(GObject *object) {
  auto *self = reinterpret_cast<GstNoesisForceIdr *>(object);
  g_free(self->stream_id);
  self->stream_id = nullptr;
  g_mutex_clear(&self->lock);
  G_OBJECT_CLASS(gst_noesis_force_idr_parent_class)->finalize(object);
}

static void gst_noesis_force_idr_init(GstNoesisForceIdr *self) {
  g_mutex_init(&self->lock);
  self->request_sequence = 0;
  self->accepted_sequence = 0;
  self->last_request_ok = FALSE;
  self->stream_id = g_strdup(kDefaultStreamId);

  gst_base_transform_set_in_place(GST_BASE_TRANSFORM(self), TRUE);
  gst_base_transform_set_passthrough(GST_BASE_TRANSFORM(self), TRUE);
}

static void gst_noesis_force_idr_class_init(GstNoesisForceIdrClass *klass) {
  auto *object_class = G_OBJECT_CLASS(klass);
  auto *element_class = GST_ELEMENT_CLASS(klass);
  auto *transform_class = GST_BASE_TRANSFORM_CLASS(klass);

  object_class->set_property = gst_noesis_force_idr_set_property;
  object_class->get_property = gst_noesis_force_idr_get_property;
  object_class->finalize = gst_noesis_force_idr_finalize;

  properties[kPropertyRequestSequence] = g_param_spec_uint(
      "request-sequence", "Request sequence",
      "Strictly monotonic sequence; each advance requests one downstream "
      "NVIDIA encoder IDR",
      0, G_MAXUINT, 0,
      static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
                               G_PARAM_EXPLICIT_NOTIFY |
                               GST_PARAM_MUTABLE_PLAYING));
  properties[kPropertyAcceptedSequence] = g_param_spec_uint(
      "accepted-sequence", "Accepted sequence",
      "Most recent request sequence accepted by the downstream event path", 0,
      G_MAXUINT, 0,
      static_cast<GParamFlags>(G_PARAM_READABLE | G_PARAM_STATIC_STRINGS));
  properties[kPropertyLastRequestOk] = g_param_spec_boolean(
      "last-request-ok", "Last request accepted",
      "Whether downstream accepted the most recent monotonic IDR request",
      FALSE,
      static_cast<GParamFlags>(G_PARAM_READABLE | G_PARAM_STATIC_STRINGS));
  properties[kPropertyStreamId] = g_param_spec_string(
      "stream-id", "Stream ID",
      "NVIDIA encoder stream identifier carried by the force-IDR event",
      kDefaultStreamId,
      static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
                               GST_PARAM_MUTABLE_READY));
  g_object_class_install_properties(object_class, kPropertyCount, properties);

  gst_element_class_add_static_pad_template(element_class, &sink_template);
  gst_element_class_add_static_pad_template(element_class, &src_template);
  gst_element_class_set_static_metadata(
      element_class, "Noesis NVIDIA force-IDR event bridge", "Filter/Video",
      "Passes buffers unchanged and emits NVIDIA's official downstream "
      "force-IDR event on monotonic requests",
      "Noesis contributors");

  transform_class->transform_ip = gst_noesis_force_idr_transform_ip;
  transform_class->passthrough_on_same_caps = TRUE;
}

static gboolean plugin_init(GstPlugin *plugin) {
  GST_DEBUG_CATEGORY_INIT(gst_noesis_force_idr_debug, "noesisforceidr", 0,
                          "Noesis NVIDIA force-IDR event bridge");
  return gst_element_register(plugin, "noesisforceidr", GST_RANK_NONE,
                              gst_noesis_force_idr_get_type());
}

GST_PLUGIN_DEFINE(GST_VERSION_MAJOR, GST_VERSION_MINOR, noesisforceidr,
                  "Noesis NVIDIA force-IDR event bridge", plugin_init,
                  PACKAGE_VERSION, "MIT", PACKAGE, "https://noesis.local")
