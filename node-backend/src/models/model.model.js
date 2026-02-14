import mongoose, { Schema } from "mongoose";

const modelSchema = new Schema({
  modelName: {
    type: String,
  },
  datasetId:
  {
    type: Schema.Types.ObjectId,
    ref: "Dataset",
    default: null,
  },
  targetColumn: {
    type: String,
    default: null
  },
  handlingMissingValueStrategy: {
    type: String,
    enum: [
      "drop_rows",
      "drop_columns",
      "mean",
      "median",
      "mode",
      "constant",
      "ffill",
      "bfill",
      "knn",
      "interpolation"
    ],
    default: null,
  },
  encodingCategoricalMethod: {
    type: String,
    enum: [
      "one_hot",
      "label",
      "ordinal",
      "binary",
      "frequency",
      "target",
      "hashing"
    ],
    default: null,
  },
  normalizationTechnique: {
    type: String,
    enum: [
      "min_max",
      "zscore",
      "robust",
      "maxabs",
      "log",
      "power_transform",
      "quantile",
      "none"
    ],
    default: null,
  },
  algorithm: {
    type: String,
    enum: [
      "logistic",
      "knn",
      "svm",
      "random_forest",
      "gradient_boosting",
      "linear_regression"
    ],
    default: null,
  },
  metrics: {
    // Classification
    accuracy: { type: Number },
    precision: { type: Number },
    recall: { type: Number },
    f1_score: { type: Number },

    // Regression
    mse: { type: Number },
    rmse: { type: Number },
    r2_score: { type: Number },
  },
  modelPath: {
    type: String
  },
  projectId: {
    type: Schema.Types.ObjectId,
    ref: "Project"
  },
  userId: {
    type: Schema.Types.ObjectId,
    ref: "User"
  }
}, { timestamps: true })


export const Model = mongoose.model("Model", modelSchema)

/*
  trainingStatus: {
    type: String,
    enum: ["pending", "training", "completed", "failed"],
    default: "pending",
  },
*/
