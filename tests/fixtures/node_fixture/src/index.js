// Mixed CommonJS + ESM imports, reaching across to TS files (extension
// probing should find lib.ts / pets.ts) and into node_modules (leftpad).
import { Animal } from "./lib";
import { Dog, Cat } from "./pets";
import { describeAll } from "./utils/helpers";
const leftpad = require("leftpad");
const scoped = require("@scoped/pkg");

class Calculator {
  constructor(base) {
    this.base = base;
  }

  describe() {
    const a = new Animal("rex");
    return a.describe();
  }
}

function run() {
  const dog = new Dog("rex");
  const cat = new Cat();
  // Direct cross-file call: describeAll lives in utils/helpers.ts and is
  // not redefined locally — Tier A linking should land it as `certain`.
  return describeAll([dog.greet("world"), cat.greet("world"), leftpad("hi", 5), scoped()]);
}

export { Calculator, run };
