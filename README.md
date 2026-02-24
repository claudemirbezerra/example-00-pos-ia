# First Module IA

Projeto Node.js 22 para testes com TensorFlow.js.

## Pré-requisitos

- **Node.js 22** (use [nvm](https://github.com/nvm-sh/nvm) para gerenciar versões: `nvm use`)

## Instalação

```bash
npm install
```

## Executando

```bash
# Script principal
npm start

# Exemplo básico com mais operações
npm test

# Modo desenvolvimento (com --watch)
npm run dev
```

## Estrutura

```
├── src/
│   ├── index.js          # Script principal com operações básicas
│   └── examples/
│       └── basic-example.js  # Exemplo mais completo
├── package.json
└── README.md
```

## TensorFlow.js no Node.js

Este projeto usa `@tensorflow/tfjs-node`, que inclui:
- Aceleração por CPU nativa
- Suporte a operações mais pesadas
- APIs completas de ML do TensorFlow

Para GPU (Linux/Windows com CUDA): use `@tensorflow/tfjs-node-gpu`.
