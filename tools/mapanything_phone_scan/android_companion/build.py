#!/usr/bin/env python3
"""Build a small, signed RoomWalk APK using an installed JDK and Android SDK."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import zipfile
import xml.etree.ElementTree as ET
from urllib.parse import urlsplit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdk', type=Path, default=os.environ.get('ANDROID_HOME'))
    parser.add_argument('--java-home', type=Path, default=os.environ.get('JAVA_HOME'))
    parser.add_argument('--keystore', type=Path, required=True)
    parser.add_argument('--password-file', type=Path, required=True)
    parser.add_argument('--server-url', required=True)
    parser.add_argument('--ca-certificate', type=Path)
    parser.add_argument('--three-root', type=Path, help='Installed Three.js used by the shared RoomWalk viewer')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.sdk or not args.java_home:
        parser.error('Set --sdk / ANDROID_HOME and --java-home / JAVA_HOME')
    parsed = urlsplit(args.server_url)
    if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in ('', '/'):
        parser.error('--server-url must be an HTTPS origin without credentials')
    root = Path(__file__).resolve().parent
    source = root / 'app/src/main'
    build = root / 'build'
    if build.exists(): shutil.rmtree(build)
    for name in ('assets', 'generated', 'classes', 'dex'): (build / name).mkdir(parents=True, exist_ok=True)
    # Package the same UI and bounded viewer module closure as the server.
    # Native capture and saved takes remain available without a connection.
    static = root.parent / 'static'
    for name in ('index.html', 'style.css', 'app.js', 'browser_capture.js', 'companion_capture.js', 'native_capture.js', 'calibration.js', 'path_comparison.js'):
        destination = build / 'assets/web/static' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(static / name, destination)
    three = args.three_root or root.parents[2] / 'oai2-fe/node_modules/three'
    for name in ('build/three.module.js', 'build/three.core.js',
                 'examples/jsm/controls/OrbitControls.js', 'examples/jsm/loaders/GLTFLoader.js',
                 'examples/jsm/utils/BufferGeometryUtils.js', 'examples/jsm/utils/SkeletonUtils.js', 'LICENSE'):
        destination = build / 'assets/web/three' / name.replace('examples/jsm/', 'examples/')
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(three / name, destination)
    shutil.copytree(source / 'res', build / 'res')
    trust = '<certificates src="system" />'
    (build/'assets/server.json').write_text(json.dumps({'url':args.server_url.rstrip('/')}))
    if args.ca_certificate:
        ca = args.ca_certificate.read_bytes()
        if b'PRIVATE KEY' in ca or b'BEGIN CERTIFICATE' not in ca or len(ca) > 65536:
            parser.error('Expected a bounded public PEM CA certificate')
        (build/'assets/roomwalk-ca.pem').write_bytes(ca)
        (build / 'res/raw').mkdir(exist_ok=True)
        (build / 'res/raw/roomwalk_ca.pem').write_bytes(ca)
        trust += '<certificates src="@raw/roomwalk_ca" />'
    (build / 'res/xml').mkdir(exist_ok=True)
    (build / 'res/xml/network_security_config.xml').write_text(
        '<?xml version="1.0" encoding="utf-8"?><network-security-config>'
        '<base-config cleartextTrafficPermitted="false"><trust-anchors>' + trust +
        '</trust-anchors></base-config></network-security-config>', encoding='utf-8')
    env = dict(os.environ, JAVA_HOME=str(args.java_home), PATH=str(args.java_home/'bin')+os.pathsep+os.environ['PATH'])
    tools = args.sdk / 'build-tools/36.0.0'
    android = args.sdk / 'platforms/android-36/android.jar'
    def run(*cmd: object) -> None:
        subprocess.run([str(c) for c in cmd], check=True, env=env)
    run(tools/'aapt2','compile','--dir',build/'res','-o',build/'resources.zip')
    run(tools/'aapt2','link','-o',build/'unsigned.apk','-I',android,'--manifest',source/'AndroidManifest.xml','-A',build/'assets','--java',build/'generated',build/'resources.zip')
    sources = sorted((source/'java').rglob('*.java')) + sorted((build/'generated').rglob('*.java'))
    run(args.java_home/'bin/javac','-source','8','-target','8','-Xlint:-options','-classpath',android,'-d',build/'classes',*sources)
    classes = sorted((build/'classes').rglob('*.class'))
    run(tools/'d8','--release','--min-api','29','--lib',android,'--output',build/'dex',*classes)
    with zipfile.ZipFile(build/'unsigned.apk','a') as z:
        for dex in sorted((build/'dex').glob('*.dex')): z.write(dex,dex.name,compress_type=zipfile.ZIP_STORED)
    run(tools/'zipalign','-P','16','-f','4',build/'unsigned.apk',build/'aligned.apk')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run(tools/'apksigner','sign','--ks',args.keystore,'--ks-key-alias','roomwalk','--ks-pass','file:'+str(args.password_file),'--out',args.output,build/'aligned.apk')
    run(tools/'apksigner','verify','--verbose','--print-certs',args.output)
    run(tools/'zipalign','-c','-P','16','4',args.output)
    digest=hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix('.apk.sha256').write_text(digest+'  '+args.output.name+'\n')
    manifest = ET.parse(source/'AndroidManifest.xml').getroot()
    print(json.dumps({'apk':str(args.output),'bytes':args.output.stat().st_size,'sha256':digest,'package':manifest.attrib['package'],'version':manifest.attrib['{http://schemas.android.com/apk/res/android}versionName']},indent=2))

if __name__ == '__main__': main()
