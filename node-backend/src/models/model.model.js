import mongoose, { Schema } from "mongoose";

const modelSchema = new Schema({
  modelName: {
    type: String,
  },
  fileName: {
    type: String,
  },
  handlingMissingValueStrategy: {
    type: String,
  },
  encodingCategoricalDataMethod: {
    type: String,
  },
  normalizationTechnique: {
    type: String,
  },
  mlModelName: {
    type: String,
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