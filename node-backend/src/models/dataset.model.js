import mongoose, { Schema } from "mongoose";

const datasetSchema = new Schema(
  {
    datasetFilePath: {
      type: String,
      required: true
    },
    originalFileName: {
      type: String,
      required: true
    },
    fileSize: {
      type: Number,
      required: true
    },
    projectId: {
      type: Schema.Types.ObjectId,
      ref: "Project"
    },
    modelId: {
      type: Schema.Types.ObjectId,
      ref: "Model"
    },
    userId: {
      type: Schema.Types.ObjectId,
      ref: "User"
    }
  },
  {
    timestamps: true
  }
)

export const Dataset = mongoose.model("Dataset", datasetSchema)