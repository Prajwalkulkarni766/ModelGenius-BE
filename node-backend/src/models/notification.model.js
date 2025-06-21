import mongoose, { Schema } from "mongoose";

const notificationSchema = new Schema({
  severity: {
    type: String,
    required: true,
    enum: ['error', 'info', 'success', 'warning'],
  },
  title: {
    type: String,
    required: true
  },
  description: {
    type: String,
    required: true
  },
  userId: {
    type: Schema.Types.ObjectId,
    ref: "User"
  }
}, { timestamps: true })


export const Notification = mongoose.model("Notification", notificationSchema)