// Type alias, enum, generic function: TS-only constructs the resolver should
// emit as definitions.

export interface Closer {
  close(): void;
}

export type Pair<T> = { first: T; second: T };

export enum Status {
  ACTIVE,
  INACTIVE,
}

export function identity<T>(x: T): T {
  return x;
}

export const FALLBACK_NAME: string = "anon";

// Uniquely-named function used by index.js — exercises cross-file call edge
// resolution (Tier A: explicit `imported_names` match against the target's
// defs).
export function describeAll(items: string[]): string {
  return items.join(", ");
}

