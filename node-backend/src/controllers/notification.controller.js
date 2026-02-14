import { Notification } from "../models/notification.model.js"
import { ApiError } from "../utils/ApiError.js"
import { ApiResponse } from "../utils/ApiResponse.js"
import { asyncHandler } from "../utils/asyncHandler.js"

const createNotification = asyncHandler(async (req, res) => {

  const { severity, title, description, userId } = req.body;

  if (!severity || !title || !description || !userId) {
    throw new ApiError(400, "All fields are required")
  }

  const notification = await Notification.create({
    severity,
    title,
    description,
    userId
  });

  return res.status(201).json(
    new ApiResponse(200, notification, "Notification generated Successfully")
  )
})

const getUserNotifications = asyncHandler(async (req, res) => {
  const { _id } = req.user;

  if (!_id) {
    return res.status(400).json(new ApiError(400, "User ID is required"));
  }

  const notifications = await Notification.find({ userId: _id }).sort({
    createdAt: -1
  })

  return res.status(200).json(
    new ApiResponse(200, notifications, "Notifications fetched Successfully")
  )
})

export {
  createNotification,
  getUserNotifications,
}
