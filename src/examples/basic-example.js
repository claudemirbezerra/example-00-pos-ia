import * as tf from '@tensorflow/tfjs-node';

async function runBasicExample() {
  console.log('=== Exemplo básico TensorFlow.js ===\n');

  // 1. Criar tensores
  const scalar = tf.scalar(42);
  const vector = tf.tensor1d([1, 2, 3, 4, 5]);
  const matrix = tf.tensor2d([[1, 2], [3, 4]]);

  console.log('Scalar:', scalar.dataSync()[0]);
  console.log('Vector:', vector.dataSync());
  console.log('Matrix:');
  matrix.print();

  // 2. Operações matemáticas
  const doubled = vector.mul(2);
  console.log('\nVector * 2:', doubled.dataSync());

  // 3. Normalização simples
  const x = tf.tensor1d([1, 2, 3, 4, 5]);
  const mean = x.mean();
  const std = tf.moments(x).variance.sqrt();
  const normalized = x.sub(mean).div(std);
  console.log('\nValores normalizados:', normalized.dataSync().map(n => n.toFixed(4)));

  // 4. Modelo linear simples (y = 2x + 1)
  const model = tf.sequential({
    layers: [
      tf.layers.dense({ inputShape: [1], units: 1 })
    ]
  });
  model.compile({ optimizer: 'sgd', loss: 'meanSquaredError' });

  const xs = tf.tensor2d([[1], [2], [3], [4]], [4, 1]);
  const ys = tf.tensor2d([[3], [5], [7], [9]], [4, 1]);

  await model.fit(xs, ys, { epochs: 100, verbose: 0 });
  
  const prediction = model.predict(tf.tensor2d([[5]], [1, 1]));
  console.log('\nPrevisão para x=5 (esperado y=11):', prediction.dataSync()[0].toFixed(2));

  // Limpar
  tf.dispose([scalar, vector, matrix, doubled, x, mean, std, normalized, xs, ys, prediction]);

  console.log('\n✅ Exemplo concluído!');
}

runBasicExample().catch(console.error);
