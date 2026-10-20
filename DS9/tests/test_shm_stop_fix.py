"""Private CPU-only same-plugin proof. No application/service/plugin selection writes."""
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'DS9/native/gst_shm_stop_fix'


@pytest.fixture(scope='module')
def private_plugin(tmp_path_factory):
    scratch = tmp_path_factory.mktemp('shm-plugin')
    selected = os.environ.get('NOESIS_SHM_TEST_PLUGIN')
    if selected:
        plugin = Path(selected).resolve(strict=True)
    else:
        subprocess.run(['bash', str(ROOT / 'DS9/scripts/build_shm_stop_fix_host.sh'), str(scratch)],
                       check=True, capture_output=True, text=True, timeout=30)
        plugin = scratch / 'libgstshm.so'
    core = Path('/usr/lib/x86_64-linux-gnu/gstreamer-1.0/libgstcoreelements.so')
    assert core.is_file()
    (scratch / core.name).symlink_to(core)
    env = os.environ.copy()
    # Scan ONLY this plugin and the standard CPU coreelements fixture, no GPU
    # plugin directories or global registry edits. These values are child-only.
    for key in ('GST_PLUGIN_PATH', 'GST_PLUGIN_SYSTEM_PATH', 'GST_REGISTRY'):
        env.pop(key, None)
    env.update(GST_PLUGIN_PATH_1_0=f'{plugin.parent}:{scratch}',
               GST_PLUGIN_SYSTEM_PATH_1_0='', GST_PLUGIN_SYSTEM_PATH='',
               GST_REGISTRY=str(scratch / 'registry.bin'))
    inspected = subprocess.run(['gst-inspect-1.0', 'shmsink'], env=env, capture_output=True,
                               text=True, check=True, timeout=10).stdout
    filename = next(line.split(None, 1)[1] for line in inspected.splitlines() if line.strip().startswith('Filename'))
    assert Path(filename).resolve() == plugin
    assert 'Version                  1.24.2' in inspected
    assert 'Noesis stop fix' in inspected
    (scratch / 'plugin.txt').write_text(inspected)
    return plugin, env, scratch


def _limit_child():
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))


def _launch(plugin, env, scratch, name, elements):
    log = scratch / f'{name}.log'
    command = ['gst-launch-1.0', '--no-position', '--gst-debug-no-color',
               f'--gst-plugin-load={plugin}', '--gst-debug=shmsink:6,GST_POLL:6', *elements]
    with log.open('xb') as output:
        result = subprocess.run(command, env=env, cwd=scratch, stdout=output,
                                stderr=subprocess.STDOUT, timeout=15, preexec_fn=_limit_child)
    assert log.stat().st_size < 1024 * 1024
    return result.returncode, log.read_text()


def test_private_plugin_normal_eos(private_plugin):
    plugin, env, scratch = private_plugin
    code, text = _launch(plugin, env, scratch, 'normal-eos', [
        'fakesrc', 'num-buffers=3', 'sizetype=2', 'sizemax=16', 'filltype=2', '!',
        'identity', 'sleep-time=200000', '!', 'shmsink', 'socket-path=normal.sock',
        'shm-size=65536', 'perms=384', 'wait-for-connection=false', 'sync=false', 'async=false'])
    assert code == 0, text[-4000:]
    assert 'Got EOS from element' in text and 'Setting pipeline to NULL' in text
    assert 'we are flushing' in text
    assert 'Failed waiting on fd activity' not in text and 'ERROR' not in text
    assert not (scratch / 'normal.sock').exists()


def test_private_plugin_genuine_open_error_remains_fatal(private_plugin):
    plugin, env, scratch = private_plugin
    code, text = _launch(plugin, env, scratch, 'genuine-error', [
        'fakesrc', 'num-buffers=1', '!', 'shmsink', 'socket-path=missing/invalid.sock',
        'wait-for-connection=false', 'sync=false', 'async=false'])
    assert code != 0
    assert 'Could not open socket' in text and 'ERROR' in text
    assert not (scratch / 'missing').exists()


def test_actual_poll_function_preserves_unrelated_errors(tmp_path):
    # Compile the ACTUAL production pollthread, replacing only the wait result
    # in this test. No copied decision predicate and no background SDK thread.
    fixture = tmp_path / 'poll.c'
    fixture.write_text(r'''
#include <errno.h>
#include "gstshmsink.h"
static GstShmSink *active;
static int stopping, failure;
static int fixture_wait(GstPoll *poll, GstClockTime timeout) {
  (void)poll; (void)timeout;
  active->stop=stopping; errno=failure; return -1;
}
#define gst_poll_wait fixture_wait
#include "gstshmsink.c"
#undef gst_poll_wait
static int check(int stop,int error,int expected) {
  GstElement *pipeline=gst_pipeline_new(NULL);
  active=g_object_new(GST_TYPE_SHM_SINK,NULL);
  gst_bin_add(GST_BIN(pipeline),GST_ELEMENT(active));
  GstBus *bus=gst_element_get_bus(pipeline);gst_bus_set_flushing(bus,FALSE);
  stopping=stop;failure=error;active->stop=FALSE;
  pollthread_func(active);
  GstMessage *message=gst_bus_pop_filtered(bus,GST_MESSAGE_ERROR);
  int actual=message!=NULL;
  if(message)gst_message_unref(message);
  gst_object_unref(bus);gst_object_unref(pipeline);
  return actual==expected;
}
int main(void) {
  gst_init(NULL,NULL);
  return !(check(1,EBUSY,0) && check(0,EBUSY,1) && check(1,EBADF,1) && check(0,EBADF,1));
}
''')
    flags = subprocess.check_output(['pkg-config', '--cflags', '--libs', 'gstreamer-base-1.0'], text=True).split()
    exe = tmp_path / 'poll'
    subprocess.run(['cc', '-std=c11', '-D_GNU_SOURCE', '-DSHM_PIPE_USE_GLIB', '-I', str(SOURCE),
                    str(fixture), str(SOURCE / 'shmpipe.c'), str(SOURCE / 'shmalloc.c'),
                    *flags, '-lrt', '-o', str(exe)], check=True, capture_output=True, text=True, timeout=15)
    subprocess.run([str(exe)], check=True, capture_output=True, text=True, timeout=5)


def test_exact_upstream_sources_and_only_narrow_delta():
    provenance = json.loads((SOURCE / 'provenance.json').read_text())
    for name, expected in provenance['files'].items():
        data = (SOURCE / name).read_bytes()
        if name == 'gstshmsink.c':
            delta = provenance['only_source_delta'].encode()
            assert data.count(delta) == 1
            data = data.replace(delta, b'', 1)
        assert hashlib.sha256(data).hexdigest() == expected['upstream_sha256']
        assert hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() == expected['upstream_git_blob']
