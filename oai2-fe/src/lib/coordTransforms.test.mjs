import assert from 'node:assert/strict';
import test from 'node:test';
import { readFile } from 'node:fs/promises';
import ts from 'typescript';

const compileModule = async (fileName, replacements = {}) => {
  const source = await readFile(new URL(`./${fileName}.ts`, import.meta.url), 'utf8');
  const compiled = ts.transpileModule(source, {
    compilerOptions: {
      module: ts.ModuleKind.ESNext,
      target: ts.ScriptTarget.ES2022,
    },
    fileName: `${fileName}.ts`,
  }).outputText;
  const rewritten = Object.entries(replacements).reduce(
    (text, [specifier, moduleUrl]) => text.replaceAll(`'./${specifier}'`, `'${moduleUrl}'`),
    compiled,
  );
  const url = `data:text/javascript;base64,${Buffer.from(rewritten).toString('base64')}`;
  return { module: await import(url), url };
};

const cameraCompiled = await compileModule('camera');
const calibrationCompiled = await compileModule('calibration', { camera: cameraCompiled.url });
const transformsCompiled = await compileModule('coordTransforms', {
  camera: cameraCompiled.url,
  calibration: calibrationCompiled.url,
});
const calibration = calibrationCompiled.module;
const transforms = transformsCompiled.module;

const kitchenExtrinsics = [
  -0.9945218953682733, -0.03745965105035022, 0.09758572731851303, 0.0,
  2.093134581417459e-17, -0.9335804264972019, -0.35836794954530016, 0.0,
  0.1045284632676535, -0.3564047724210337, 0.9284661752387182, 0.0,
  13.885812752498177, 2.7020906876037474, -0.7184407653953737, 1.0,
];

const familyExtrinsics = [
  0.9243373284430696, -0.07202118346826888, 0.37471783034512596, 0.0,
  -0.0015810945074533217, -0.9827403339356952, -0.18498361061487914, 0.0,
  0.38157306428374516, 0.170394792138107, -0.9084974471209725, 0.0,
  -18.932472331975617, 0.910475677821805, 6.136613144161239, 1.0,
];

const kitchenBinding = [
  1.0, 0.0, 0.0, 0.0,
  0.0, 1.0, 0.0, 0.0,
  0.0, 0.0, 1.0, 0.0,
  0.0, 0.0, 0.0, 1.0,
];

const familyBinding = [
  0.9467470066425219, -0.09059934064225462, -0.30896903548513316, 0.0,
  0.0649604489274947, 0.9936095819714693, -0.09230459733740415, 0.0,
  0.31535732984752884, 0.06731833397872138, 0.9465822713434718, 0.0,
  -3.1508635068958624, 0.17446172385443262, 5.462715819965061, 1.0,
];

calibration.setCalibration({
  cameras: {
    E: {
      kitchen: kitchenExtrinsics,
      'family-room': familyExtrinsics,
    },
    frame_bindings: {
      kitchen: { target_from_calibration_col_major: kitchenBinding },
      'family-room': { target_from_calibration_col_major: familyBinding },
    },
  },
});

const near = (actual, expected, tolerance = 1e-6) => {
  assert.ok(
    Math.abs(actual - expected) <= tolerance,
    `expected ${actual} to be within ${tolerance} of ${expected}`,
  );
};

test('Kitchen ground projection anchors a pitched camera at its floor origin', () => {
  const projected = transforms.projectWorldPointToCameraLocal(
    'kitchen',
    13.98107375623384,
    0,
    0.17862329791760775,
  );

  assert.ok(projected);
  near(projected.x, 0);
  near(projected.y, 0);
});

test('Family ground projection preserves the same horizontal origin contract', () => {
  const projected = transforms.projectWorldPointToCameraLocal(
    'family-room',
    15.419562592882802,
    0,
    12.530025503302328,
  );

  assert.ok(projected);
  near(projected.x, 0);
  near(projected.y, 0);
});

test('ground projection follows horizontal camera axes independently of pitch and height', () => {
  const kitchen = transforms.projectWorldPointToCameraLocal(
    'kitchen',
    14.08560222,
    0,
    1.17314519,
  );
  const family = transforms.projectWorldPointToCameraLocal(
    'family-room',
    15.478131803809735,
    0,
    11.531742152979687,
  );

  assert.ok(kitchen);
  assert.ok(family);
  near(kitchen.x, 0, 1e-6);
  near(kitchen.y, 1, 1e-6);
  near(family.x, 0, 1e-6);
  near(family.y, 1, 1e-6);
});

test('revision-bound projection fails closed when a camera binding is absent', () => {
  calibration.setCalibration({
    cameras: {
      E: { kitchen: kitchenExtrinsics },
      frame_bindings: {},
    },
  });

  assert.equal(
    transforms.projectWorldPointToCameraLocal(
      'kitchen',
      13.98107375623384,
      0,
      0.17862329791760775,
    ),
    null,
  );
});
