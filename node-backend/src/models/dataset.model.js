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
    columns: [{
      type: String,
    }],
    projectId: {
      type: Schema.Types.ObjectId,
      ref: "Project"
    },
  },
  {
    timestamps: true
  }
)

export const Dataset = mongoose.model("Dataset", datasetSchema)