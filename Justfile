set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

# Diretórios de build. Podem ser sobrescritos, por exemplo:
# just native BUILD_DIR=/tmp/nnw-build
BUILD_DIR := "build"
WASM_BUILD_DIR := "build-wasm"
PYTHON_VENV := ".venv"
PYTHON := PYTHON_VENV + "/bin/python"

default:
    @just --list

# Configura e compila a biblioteca nativa e os exemplos.
native:
    cmake -S . -B {{BUILD_DIR}} -DBUILD_EXAMPLES=ON
    cmake --build {{BUILD_DIR}} --config Release --parallel

# Configura e compila somente a biblioteca nativa.
native-lib:
    cmake -S . -B {{BUILD_DIR}} -DBUILD_EXAMPLES=OFF
    cmake --build {{BUILD_DIR}} --config Release --parallel

# Executa um exemplo já compilado. Uso: just run-example xor
run-example example:
    ./{{BUILD_DIR}}/{{example}}

# Configura e compila os artefatos WebAssembly.
wasm:
    emcmake cmake -S . -B {{WASM_BUILD_DIR}} -DBUILD_EXAMPLES=OFF
    cmake --build {{WASM_BUILD_DIR}} --config Release --parallel

# Compila o pacote TypeScript usando o runtime WebAssembly existente.
typescript:
    cd bindings/typescript && pnpm install --frozen-lockfile && pnpm build

# Compila o pacote TypeScript e copia os artefatos WebAssembly para dist.
typescript-full:
    cd bindings/typescript && pnpm install --frozen-lockfile && pnpm build:full

# Cria a virtualenv Python local, caso ainda não exista.
python-venv:
    if [ ! -x "{{PYTHON}}" ]; then python3 -m venv {{PYTHON_VENV}}; fi

# Cria o wheel Python usando exclusivamente a virtualenv do projeto.
python-package: python-venv
    {{PYTHON}} -m pip install --upgrade build
    {{PYTHON}} -m build -w bindings/python

# Remove apenas os diretórios de build gerados pelo projeto.
clean:
    rm -rf {{BUILD_DIR}} {{WASM_BUILD_DIR}}

# Mostra os comandos disponíveis.
list:
    @just --list
