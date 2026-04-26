// Inheritance under test:
//   - Dog extends Animal (class extends, single base)
//   - Dog implements Greeter (TS implements, single name)
//   - Cat extends Animal implements Greeter, Closer (multiple implements)
// Imports use the tsconfig path alias `@app/*`. Cross-file resolution should
// connect Animal/Greeter to lib.ts and Closer to utils/helpers.ts.

import { Animal, Greeter } from "@app/lib";
import { Closer, FALLBACK_NAME } from "@app/utils/helpers";

export class Dog extends Animal implements Greeter {
  greet(n: string): string {
    return `Woof, ${n}`;
  }
}

export class Cat extends Animal implements Greeter, Closer {
  constructor(name: string = FALLBACK_NAME) {
    super(name);
  }

  greet(n: string): string {
    return `Meow, ${n}`;
  }

  close(): void {
    // no-op
  }
}
