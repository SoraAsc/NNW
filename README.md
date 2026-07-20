# NNW

NNW is a C++17 library for feed-forward neural networks and reinforcement learning. It provides a stable C API and bindings for TypeScript/WebAssembly, Unity, C#, and Python.

## Features

- Dense neural networks with linear, ReLU, sigmoid, and tanh activations.
- Supervised training with SGD or AdamW and mean squared error.
- PPO with discrete, continuous, multi-discrete, and multi-binary action spaces.
- Tabular Q-learning.
- Model checkpoints and parameter import/export.
- Native and WebAssembly builds.

## Native build

Requirements:

- CMake 3.15 or newer
- A C++17 compiler

```sh
cmake -S . -B build -DBUILD_EXAMPLES=ON
cmake --build build --config Release
```

## WebAssembly build

With Emscripten available in the current shell:

```sh
emcmake cmake -S . -B build-wasm -DBUILD_EXAMPLES=OFF
cmake --build build-wasm --config Release
```

The build produces a legacy runtime (`nn.js` and `nn.wasm`) and an ES module runtime (`nn_esm.mjs` and `nn_esm.wasm`).

## TypeScript

```ts
import { createNNW } from "nnw";

const nnw = await createNNW();
const model = nnw
  .createModel(2)
  .addDense(4, "tanh")
  .addDense(1, "sigmoid");

const trainer = nnw.createTrainer(model, {
  optimizer: "adamw",
  learningRate: 0.03,
  batchSize: 4,
});

const inputs = [0, 0, 0, 1, 1, 0, 1, 1];
const targets = [0, 1, 1, 0];

for (let epoch = 0; epoch < 500; epoch++) {
  trainer.train(inputs, targets);
}

console.log(model.predict(inputs));

trainer.dispose();
model.dispose();
```

Build the TypeScript package after compiling the WebAssembly runtime:

```sh
cd bindings/typescript
pnpm install
pnpm build:full
```

## Examples

The [`examples`](examples) directory contains native examples for supervised learning, PPO, and Q-learning. [Neural Playground](https://soraasc.github.io/NeuralPlayground/) provides interactive browser demonstrations built with NNW.

## Releases

Tags matching `v*` trigger the release workflow, which builds native libraries, WebAssembly artifacts, the TypeScript package, Python wheel, C# package, and Unity package.
