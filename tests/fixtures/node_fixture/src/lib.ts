// Shared types for the fixture: an interface, an interface that extends, and a
// concrete class. Imported by pets.ts via the tsconfig path alias `@app/lib`,
// and by index.js via a relative `./lib` specifier (extension probing should
// land on this .ts file).

export interface Greeter {
  greet(name: string): string;
}

export interface Bilingual extends Greeter {
  greetIn(lang: string, name: string): string;
}

export class Animal {
  constructor(public name: string) {}

  describe(): string {
    return `Animal(${this.name})`;
  }
}
