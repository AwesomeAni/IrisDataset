const tf = require("@tensorflow/tfjs");
const iris = require("./iris.json");
const irisTesting = require("./iris-testing.json");

// Load the data
const trainingData = tf.tensor2d(iris.map((item) => [
  item.sepal_length, item.sepal_width, item.petal_length, item.petal_width
]));
const testingData = tf.tensor2d(irisTesting.map((item) => [
  item.sepal_length, item.sepal_width, item.petal_length, item.petal_width
]));
const outputData = tf.tensor2d(iris.map((item) => [
  item.species === "setosa" ? 1 : 0,
  item.species === "virginica" ? 1 : 0,
  item.species === "versicolor" ? 1 : 0
]))


// Build a model
const model = tf.sequential();
model.add(tf.layers.dense({units: 5, inputShape: [4]}))
model.add(tf.layers.dense({units: 3, activation: "sigmoid"}))
model.add(tf.layers.dense({units: 3, activation: "sigmoid"}))
model.compile({optimizer: tf.train.adam(.08), loss: "meanSquaredError", metrics: ["accuracy"]})
// Train the model
const startTime = Date.now();
model.fit(trainingData, outputData, {epochs:100})
// Test the model
.then((history) => {
  console.log(history)
  model.predict(testingData).print() // Expected output [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
})
