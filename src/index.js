import tf from '@tensorflow/tfjs-node';

async function trainModel(inputXs, outputYs) {
    const model = tf.sequential();

    //Primeira camada da rede neural

    //80 neuronios = quanto mais neuronios, mais complexo o modelo, mas mais lento o treinamento
    // a Relu age com um filtro:
    // como se ela deixasse apenas os dados interessantes para o modelo
    // se for zero ou negativo, ela tira o dado
    model.add(tf.layers.dense({ units: 80, inputShape: [7], activation: 'relu' }));

    // saida: 3 neuronios = premium, medium, basic (um para cada categoria)
    // activation: softmax = a saida é uma probabilidade de cada categoria
    model.add(tf.layers.dense({units: 3, activation: 'softmax' }));

    // Compilando o modelo
    // optimizer Adam ( Adaptive moment Estimation)
    // um treinador pessoal moderno para redes neurais
    // ajusta os pesos de forma eficiente e adaptativa
    // aprender com historico de erros e acertos

    // loss: categoricalCrossentropy
    // Ele compara o que o modelo "acha" (os scores de cada categoria)
    // com a resposta certa
    // a categoria premium será sempre [1, 0, 0]

    // quanto mais distante da previsão do modelo da resposta correta
    // maior o erro (loss)
    // Exemplo classico: classificação de imagens, recomendação, categorização de
    // usuário
    // qualquer coisa em que a resposta certa é "apenas uma entre várias possíveis"

    model.compile({ optimizer: 'adam', 
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    // Treinando o modelo
    // epochs: quantas vezes o modelo vai ver os dados
    // shuffle: se vai embaralhar os dados
    // verbose: desabilita o log interno (e usa somente o callback)
    await model.fit(inputXs, 
        outputYs, 
        { 
            verbose: 0,
            epochs: 100,
            shuffle: true,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    //console.log(`Epoch ${epoch} - Loss: ${logs.loss}`);
                }
            }
        });

    return model;
}

async function predict(model, pessoaNormalizada) {
    // transformar o array js para o tensor
    const tfInput = tf.tensor2d(pessoaNormalizada)

    // Faz a predição
    const pred = model.predict(tfInput)
    const predArray = await pred.array()
    console.log(predArray)

    return predArray[0].map((prob, index) => ({ prob, index }));
}

// Exemplo de pessoas para treino (cada pessoa com idade, cor e localização)
// const pessoas = [
//     { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo" },
//     { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio" },
//     { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba" }
// ];

// Vetores de entrada com valores já normalizados e one-hot encoded
// Ordem: [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
// const tensorPessoas = [
//     [0.33, 1, 0, 0, 1, 0, 0], // Erick
//     [0, 0, 1, 0, 0, 1, 0],    // Ana
//     [1, 0, 0, 1, 0, 0, 1]     // Carlos
// ]

// Usamos apenas os dados numéricos, como a rede neural só entende números.
// tensorPessoasNormalizado corresponde ao dataset de entrada do modelo.
const tensorPessoasNormalizado = [
    [0.33, 1, 0, 0, 1, 0, 0], // Erick
    [0, 0, 1, 0, 0, 1, 0],    // Ana
    [1, 0, 0, 1, 0, 0, 1]     // Carlos
]

// Labels das categorias a serem previstas (one-hot encoded)
// [premium, medium, basic]
const labelsNomes = ["premium", "medium", "basic"]; // Ordem dos labels
const tensorLabels = [
    [1, 0, 0], // premium - Erick
    [0, 1, 0], // medium - Ana
    [0, 0, 1]  // basic - Carlos
];

// Criamos tensores de entrada (xs) e saída (ys) para treinar o modelo
const inputXs = tf.tensor2d(tensorPessoasNormalizado)
const outputYs = tf.tensor2d(tensorLabels)

// Quanto mais dados, melhor o modelo
const model = await trainModel(inputXs, outputYs)

const pessoa = { nome: "ze", idade: 28, cor: "verde", localizacao: "Curitiba" }
// Ordem: [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
const pessoaNormalizada = [
    [0.2, 1, 0, 0, 1, 0, 0]
]

const predictions = await predict(model, pessoaNormalizada)
const result = predictions
.sort((a, b) => b.prob - a.prob)
.map(p => `${labelsNomes[p.index]} (${(p.prob * 100).toFixed(2)}%`)

console.log(result)