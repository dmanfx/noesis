const BASE_PALETTE = [
  '#042aff', '#0bdbeb', '#f3f3f3', '#00dfb7', '#111f68',
  '#ff6fdd', '#ff444f', '#cced00', '#00f344', '#bd00ff',
  '#00b4ff', '#dd00ba', '#00ffff', '#26c000', '#01ffb3',
  '#7d24ff', '#7b0068', '#ff1b6c', '#fc6d2f', '#a2ff0b',
];

export const ADE20K_LABELS = `wall
building
sky
floor
tree
ceiling
road
bed
windowpane
grass
cabinet
sidewalk
person
earth
door
table
mountain
plant
curtain
chair
car
water
painting
sofa
shelf
house
sea
mirror
rug
field
armchair
seat
fence
desk
rock
wardrobe
lamp
bathtub
railing
cushion
base
box
column
signboard
chest of drawers
counter
sand
sink
skyscraper
fireplace
refrigerator
grandstand
path
stairs
runway
case
pool table
pillow
screen door
stairway
river
bridge
bookcase
blind
coffee table
toilet
flower
book
hill
bench
countertop
stove
palm
kitchen island
computer
swivel chair
boat
bar
arcade machine
hovel
bus
towel
light
truck
tower
chandelier
awning
streetlight
booth
television receiver
airplane
dirt track
apparel
pole
land
bannister
escalator
ottoman
bottle
buffet
poster
stage
van
ship
fountain
conveyor belt
canopy
washer
plaything
swimming pool
stool
barrel
basket
waterfall
tent
bag
minibike
cradle
oven
ball
food
step
tank
trade name
microwave
pot
animal
bicycle
lake
dishwasher
screen
blanket
sculpture
hood
sconce
vase
traffic light
tray
ashcan
fan
pier
crt screen
plate
monitor
bulletin board
shower
radiator
glass
clock
flag`.split('\n');

const byteToHex = (value) => Math.round(value).toString(16).padStart(2, '0');

const hsvToRgb = (hue, saturation, value) => {
  const chroma = value * saturation;
  const sector = hue / 60;
  const intermediate = chroma * (1 - Math.abs((sector % 2) - 1));
  let red = 0;
  let green = 0;
  let blue = 0;

  if (sector < 1) [red, green, blue] = [chroma, intermediate, 0];
  else if (sector < 2) [red, green, blue] = [intermediate, chroma, 0];
  else if (sector < 3) [red, green, blue] = [0, chroma, intermediate];
  else if (sector < 4) [red, green, blue] = [0, intermediate, chroma];
  else if (sector < 5) [red, green, blue] = [intermediate, 0, chroma];
  else [red, green, blue] = [chroma, 0, intermediate];

  const match = value - chroma;
  return [
    (red + match) * 255,
    (green + match) * 255,
    (blue + match) * 255,
  ];
};

export function semanticColor(classId) {
  const normalizedId = Math.max(0, Math.trunc(Number(classId) || 0));
  if (normalizedId < BASE_PALETTE.length) return BASE_PALETTE[normalizedId];

  const hue = (normalizedId * 137.507764 + 29) % 360;
  const saturation = 0.66 + ((normalizedId % 4) * 0.07);
  const value = 0.88 + ((normalizedId % 3) * 0.04);
  return `#${hsvToRgb(hue, saturation, Math.min(value, 0.96)).map(byteToHex).join('')}`;
}

export function semanticColorRgb(classId) {
  const hex = semanticColor(classId);
  return [
    Number.parseInt(hex.slice(1, 3), 16),
    Number.parseInt(hex.slice(3, 5), 16),
    Number.parseInt(hex.slice(5, 7), 16),
  ];
}

export function summarizeClassIds(classIds) {
  if (!classIds || typeof classIds.length !== 'number' || classIds.length === 0) return [];
  const counts = new Map();
  for (let index = 0; index < classIds.length; index += 1) {
    const classId = Number(classIds[index]);
    counts.set(classId, (counts.get(classId) || 0) + 1);
  }
  return [...counts.entries()]
    .map(([classId, count]) => ({ classId, count, fraction: count / classIds.length }))
    .sort((left, right) => right.count - left.count || left.classId - right.classId);
}

export function toggleSemanticClass(selectedClassIds, classId) {
  const selected = new Set(selectedClassIds || []);
  if (selected.has(classId)) selected.delete(classId);
  else selected.add(classId);
  return [...selected];
}
