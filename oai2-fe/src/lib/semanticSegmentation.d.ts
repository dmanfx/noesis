export const ADE20K_LABELS: readonly string[];

export type SemanticClassSummary = {
  classId: number;
  count: number;
  fraction: number;
};

export function semanticColor(classId: number): string;
export function semanticColorRgb(classId: number): [number, number, number];
export function summarizeClassIds(classIds: ArrayLike<number> | null | undefined): SemanticClassSummary[];
export function toggleSemanticClass(selectedClassIds: readonly number[], classId: number): number[];
