import { Project } from "../models/project.model.js"
import { ApiError } from "../utils/ApiError.js"
import { ApiResponse } from "../utils/ApiResponse.js"
import { asyncHandler } from "../utils/asyncHandler.js"
import { Model } from "../models/model.model.js"
import { Dataset } from '../models/dataset.model.js';
import fs from 'fs/promises';
import path from 'path';

const getModels = asyncHandler(async (req, res) => {

  const { projectId } = req.params;

  const models = await Model.find({ projectId })

  return res.status(200).json(
    new ApiResponse(200, models, "Models fetched Successfully")
  )
})

const getModel = asyncHandler(async (req, res) => {

  const { modelId } = req.params;

  const model = await Model.findById(modelId)

  return res.status(200).json(
    new ApiResponse(200, model, "Models found Successfully")
  )
})

const createModel = asyncHandler(async (req, res) => {

  const { projectId } = req.params;
  const { modelName } = req.body;

  if (!modelName) {
    throw new ApiError(400, "All fields are required")
  }

  const model = await Model.create({
    modelName,
    projectId,
    userId: req.user._id
  });

  return res.status(201).json(
    new ApiResponse(200, model, "Model generated Successfully")
  )
})

const updateModel = asyncHandler(async (req, res) => {

  const { projectId, modelId } = req.params;

  const project = await Project.findById(projectId);
  if (!project) {
    throw new ApiError(404, "Project not found.");
  }

  const model = await Model.findOne({ _id: modelId, projectId });
  if (!model) {
    throw new ApiError(404, "Model not found or does not belong to this project.");
  }

  Object.keys(req.body).forEach((key) => {
    model[key] = req.body[key];
  });

  await model.save();

  return res.status(200).json(
    new ApiResponse(200, model, "Model updated successfully.")
  );
})

const deleteModel = asyncHandler(async (req, res) => {

  const { projectId, modelId } = req.params;

  const project = await Project.findById(projectId);
  if (!project) {
    throw new ApiError(404, "Project not found.");
  }

  const model = await Model.findByIdAndDelete(modelId);
  if (!model) {
    throw new ApiError(404, "Model not found.");
  }

  const datasets = await Dataset.find({ modelId });

  await Dataset.deleteMany({ modelId });

  for (const dataset of datasets) {
    try {
      const filePath = path.resolve(dataset.datasetFilePath);
      await fs.unlink(filePath);
    } catch (error) {
      console.warn(`Failed to delete file at ${dataset.datasetFilePath}:`, error.message);
    }
  }

  return res.status(204).json(
    new ApiResponse(204, model, "Model deleted Successfully")
  )
})

export {
  getModels,
  getModel,
  createModel,
  updateModel,
  deleteModel
}
