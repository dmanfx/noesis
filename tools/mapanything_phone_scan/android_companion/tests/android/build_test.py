#!/usr/bin/env python3
"""Build a separate Android instrumentation APK against the current companion build."""
import argparse,os,subprocess,zipfile
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
for name in ('sdk','java-home','keystore','password-file','output'):p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args();root=Path(__file__).resolve().parent;production=root.parents[1]/'build/classes';build=a.output.parent/'instrumentation-build'
for child in ('classes','dex'): (build/child).mkdir(parents=True,exist_ok=True)
sdkjar=a.sdk/'platforms/android-36/android.jar';tools=a.sdk/'build-tools/36.0.0';env=dict(os.environ,JAVA_HOME=str(a.java_home),PATH=str(a.java_home/'bin')+os.pathsep+os.environ['PATH'])
def run(*cmd):subprocess.run(list(map(str,cmd)),check=True,env=env)
run(tools/'aapt2','link','-o',build/'unsigned.apk','-I',sdkjar,'--manifest',root/'AndroidManifest.xml')
run(a.java_home/'bin/javac','-source','8','-target','8','-Xlint:-options','-cp',str(sdkjar)+os.pathsep+str(production),'-d',build/'classes',*sorted(root.glob('*.java')))
run(tools/'d8','--min-api','29','--lib',sdkjar,'--classpath',production,'--output',build/'dex',*sorted((build/'classes').rglob('*.class')))
with zipfile.ZipFile(build/'unsigned.apk','a') as z:z.write(build/'dex/classes.dex','classes.dex',compress_type=zipfile.ZIP_STORED)
run(tools/'zipalign','-P','16','-f','4',build/'unsigned.apk',build/'aligned.apk')
run(tools/'apksigner','sign','--ks',a.keystore,'--ks-key-alias','roomwalk','--ks-pass','file:'+str(a.password_file),'--out',a.output,build/'aligned.apk')
run(tools/'apksigner','verify',a.output)
print(a.output)
