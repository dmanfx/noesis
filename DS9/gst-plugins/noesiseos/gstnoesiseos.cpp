// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Noesis contributors

#include <gst/base/gstbasetransform.h>
#include <gst/gst.h>

#ifndef PACKAGE
#define PACKAGE "noesis"
#endif

#ifndef PACKAGE_VERSION
#define PACKAGE_VERSION "1.0.0"
#endif

#ifndef NOESIS_DEEPSTREAM_MAJOR
#define NOESIS_DEEPSTREAM_MAJOR "unknown"
#endif

#define NOESIS_EOS_DESCRIPTION \
  "Noesis orderly EOS control bridge (DS" NOESIS_DEEPSTREAM_MAJOR ")"

GST_DEBUG_CATEGORY_STATIC(gst_noesis_eos_debug);
#define GST_CAT_DEFAULT gst_noesis_eos_debug

namespace {

enum PropertyId : guint {
  kPropertyNone = 0,
  kPropertyRequestSequence,
  kPropertyAcceptedSequence,
  kPropertyLastRequestOk,
  kPropertyCount,
};

GParamSpec *properties[kPropertyCount] = {};

}  // namespace

typedef struct _GstNoesisEos {
  GstBaseTransform parent;

  GMutex lock;
  guint request_sequence;
  guint accepted_sequence;
  gboolean last_request_ok;
  gint eos_accepted;
  gboolean request_pending;
} GstNoesisEos;

typedef struct _GstNoesisEosClass {
  GstBaseTransformClass parent_class;
} GstNoesisEosClass;

G_DEFINE_TYPE(GstNoesisEos, gst_noesis_eos, GST_TYPE_BASE_TRANSFORM)

typedef struct _NoesisEosRequest {
  GstNoesisEos *self;
  guint sequence;
} NoesisEosRequest;

static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE(
    "sink", GST_PAD_SINK, GST_PAD_ALWAYS, GST_STATIC_CAPS_ANY);

static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE(
    "src", GST_PAD_SRC, GST_PAD_ALWAYS, GST_STATIC_CAPS_ANY);

static gpointer gst_noesis_eos_request_worker(gpointer data) {
  auto *request = static_cast<NoesisEosRequest *>(data);
  GstNoesisEos *self = request->self;
  const guint requested_sequence = request->sequence;

  GstEvent *event = gst_event_new_eos();
  gboolean request_ok = FALSE;
  if (event != nullptr) {
    // gst_pad_push_event() consumes the event regardless of the result. This
    // may block while asynchronous downstream elements drain, which is why it
    // must never run on the Service Maker/GObject property-setter thread.
    request_ok = gst_pad_push_event(GST_BASE_TRANSFORM_SRC_PAD(self), event);
  }

  guint previous_accepted_sequence = 0;
  gboolean previous_request_ok = FALSE;
  g_mutex_lock(&self->lock);
  previous_accepted_sequence = self->accepted_sequence;
  previous_request_ok = self->last_request_ok;
  self->request_pending = FALSE;
  self->last_request_ok = request_ok;
  if (request_ok) {
    self->accepted_sequence = requested_sequence;
    GST_INFO_OBJECT(self, "accepted orderly EOS request sequence=%u",
                    requested_sequence);
  } else {
    g_atomic_int_set(&self->eos_accepted, FALSE);
    GST_ERROR_OBJECT(self,
                     "downstream rejected orderly EOS request sequence=%u",
                     requested_sequence);
  }
  g_mutex_unlock(&self->lock);

  if (request_ok && previous_accepted_sequence != requested_sequence) {
    g_object_notify_by_pspec(G_OBJECT(self),
                             properties[kPropertyAcceptedSequence]);
  }
  if (previous_request_ok != request_ok) {
    g_object_notify_by_pspec(G_OBJECT(self),
                             properties[kPropertyLastRequestOk]);
  }

  g_object_unref(self);
  g_free(request);
  return nullptr;
}

static void gst_noesis_eos_set_property(GObject *object, guint property_id,
                                        const GValue *value,
                                        GParamSpec *param_spec) {
  auto *self = reinterpret_cast<GstNoesisEos *>(object);

  switch (property_id) {
    case kPropertyRequestSequence: {
      const guint requested_sequence = g_value_get_uint(value);
      gboolean previous_request_ok = FALSE;

      g_mutex_lock(&self->lock);
      if (requested_sequence <= self->request_sequence) {
        GST_WARNING_OBJECT(
            self,
            "ignoring non-monotonic EOS request sequence %u (current=%u)",
            requested_sequence, self->request_sequence);
        g_mutex_unlock(&self->lock);
        return;
      }
      if (self->request_pending) {
        GST_WARNING_OBJECT(
            self,
            "ignoring EOS request sequence %u while sequence %u is pending",
            requested_sequence, self->request_sequence);
        g_mutex_unlock(&self->lock);
        return;
      }

      self->request_sequence = requested_sequence;
      previous_request_ok = self->last_request_ok;
      self->request_pending = TRUE;
      g_atomic_int_set(&self->eos_accepted, TRUE);

      auto *request = g_new0(NoesisEosRequest, 1);
      request->self =
          reinterpret_cast<GstNoesisEos *>(g_object_ref(G_OBJECT(self)));
      request->sequence = requested_sequence;
      GError *thread_error = nullptr;
      GThread *worker = g_thread_try_new("noesis-eos", gst_noesis_eos_request_worker,
                                         request, &thread_error);
      if (worker == nullptr) {
        self->request_pending = FALSE;
        self->last_request_ok = FALSE;
        g_atomic_int_set(&self->eos_accepted, FALSE);
        GST_ERROR_OBJECT(self, "failed to start orderly EOS worker: %s",
                         thread_error != nullptr ? thread_error->message
                                                 : "unknown error");
        g_clear_error(&thread_error);
        g_object_unref(request->self);
        g_free(request);
      } else {
        g_thread_unref(worker);
      }
      g_mutex_unlock(&self->lock);

      g_object_notify_by_pspec(object, properties[kPropertyRequestSequence]);
      if (worker == nullptr && previous_request_ok) {
        g_object_notify_by_pspec(object, properties[kPropertyLastRequestOk]);
      }
      return;
    }
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID(object, property_id, param_spec);
      return;
  }
}

static void gst_noesis_eos_get_property(GObject *object, guint property_id,
                                        GValue *value,
                                        GParamSpec *param_spec) {
  auto *self = reinterpret_cast<GstNoesisEos *>(object);

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
    default:
      g_mutex_unlock(&self->lock);
      G_OBJECT_WARN_INVALID_PROPERTY_ID(object, property_id, param_spec);
      return;
  }
  g_mutex_unlock(&self->lock);
}

static GstFlowReturn gst_noesis_eos_transform_ip(GstBaseTransform *transform,
                                                 GstBuffer *) {
  auto *self = reinterpret_cast<GstNoesisEos *>(transform);
  if (g_atomic_int_get(&self->eos_accepted)) {
    // The buffer is neither mapped nor copied, and BaseTransform will not push
    // it into a downstream pad that has already accepted EOS.
    return GST_BASE_TRANSFORM_FLOW_DROPPED;
  }
  // Before EOS, buffers and NVMM surfaces pass through untouched.
  return GST_FLOW_OK;
}

static void gst_noesis_eos_finalize(GObject *object) {
  auto *self = reinterpret_cast<GstNoesisEos *>(object);
  g_mutex_clear(&self->lock);
  G_OBJECT_CLASS(gst_noesis_eos_parent_class)->finalize(object);
}

static void gst_noesis_eos_init(GstNoesisEos *self) {
  g_mutex_init(&self->lock);
  self->request_sequence = 0;
  self->accepted_sequence = 0;
  self->last_request_ok = FALSE;
  g_atomic_int_set(&self->eos_accepted, FALSE);
  self->request_pending = FALSE;

  gst_base_transform_set_in_place(GST_BASE_TRANSFORM(self), TRUE);
  gst_base_transform_set_passthrough(GST_BASE_TRANSFORM(self), TRUE);
}

static void gst_noesis_eos_class_init(GstNoesisEosClass *klass) {
  auto *object_class = G_OBJECT_CLASS(klass);
  auto *element_class = GST_ELEMENT_CLASS(klass);
  auto *transform_class = GST_BASE_TRANSFORM_CLASS(klass);

  object_class->set_property = gst_noesis_eos_set_property;
  object_class->get_property = gst_noesis_eos_get_property;
  object_class->finalize = gst_noesis_eos_finalize;

  properties[kPropertyRequestSequence] = g_param_spec_uint(
      "request-sequence", "Request sequence",
      "Strictly monotonic sequence; each advance requests one downstream EOS",
      0, G_MAXUINT, 0,
      static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
                               G_PARAM_EXPLICIT_NOTIFY |
                               GST_PARAM_MUTABLE_PLAYING));
  properties[kPropertyAcceptedSequence] = g_param_spec_uint(
      "accepted-sequence", "Accepted sequence",
      "Most recent EOS request sequence accepted by the downstream event path",
      0, G_MAXUINT, 0,
      static_cast<GParamFlags>(G_PARAM_READABLE | G_PARAM_STATIC_STRINGS));
  properties[kPropertyLastRequestOk] = g_param_spec_boolean(
      "last-request-ok", "Last request accepted",
      "Whether downstream accepted the most recent monotonic EOS request",
      FALSE,
      static_cast<GParamFlags>(G_PARAM_READABLE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_properties(object_class, kPropertyCount, properties);

  gst_element_class_add_static_pad_template(element_class, &sink_template);
  gst_element_class_add_static_pad_template(element_class, &src_template);
  gst_element_class_set_static_metadata(
      element_class, NOESIS_EOS_DESCRIPTION, "Filter/Video",
      "Passes buffers unchanged and injects standard EOS for orderly pipeline "
      "quiescence",
      "Noesis contributors");

  transform_class->transform_ip = gst_noesis_eos_transform_ip;
  transform_class->passthrough_on_same_caps = TRUE;
  transform_class->transform_ip_on_passthrough = TRUE;
}

static gboolean plugin_init(GstPlugin *plugin) {
  GST_DEBUG_CATEGORY_INIT(gst_noesis_eos_debug, "noesiseos", 0,
                          NOESIS_EOS_DESCRIPTION);
  return gst_element_register(plugin, "noesiseos", GST_RANK_NONE,
                              gst_noesis_eos_get_type());
}

GST_PLUGIN_DEFINE(GST_VERSION_MAJOR, GST_VERSION_MINOR, noesiseos,
                  NOESIS_EOS_DESCRIPTION, plugin_init, PACKAGE_VERSION, "MIT",
                  PACKAGE, "https://noesis.local")
