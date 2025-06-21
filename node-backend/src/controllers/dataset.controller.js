import { Dataset } from "../models/dataset.model.js"
import { Project } from "../models/project.model.js"
import { Model } from "../models/model.model.js"
import { ApiError } from "../utils/ApiError.js"
import { ApiResponse } from "../utils/ApiResponse.js"
import { asyncHandler } from "../utils/asyncHandler.js"

const getDatasets = asyncHandler(async (req, res) => {
  const { projectId, modelId } = req.params;

  const project = await Project.findById(projectId);
  if (!project) {
    throw new ApiError(404, "Project not found.");
  }

  const model = await Model.findOne({ _id: modelId, projectId });
  if (!model) {
    throw new ApiError(404, "Model not found or does not belong to this project.");
  }

  const dataset = await Dataset.find({ modelId }).select("-datasetFilePath");

  return res.status(200).json(
    new ApiResponse(200, dataset, "Dataset fetched Successfully")
  )
})

const createDataset = asyncHandler(async (req, res) => {

  const { projectId, modelId } = req.params;

  const project = await Project.findById(projectId);
  if (!project) {
    throw new ApiError(404, "Project not found.");
  }

  const model = await Model.findOne({ _id: modelId, projectId });
  if (!model) {
    throw new ApiError(404, "Model not found or does not belong to this project.");
  }

  const uploadedFiles = req.files?.datasetFile;

  if (!uploadedFiles || uploadedFiles.length === 0) {
    throw new ApiError(400, "No dataset files uploaded.");
  }

  // Create a dataset entry for each uploaded file
  const createdDatasets = await Promise.all(
    uploadedFiles.map(file => {
      return Dataset.create({
        datasetFilePath: file.path,
        originalFileName: file.originalname,
        fileSize: file.size,
        projectId,
        modelId
      });
    })
  );

  return res.status(201).json(
    new ApiResponse(200, createdDatasets, `${createdDatasets.length} dataset file(s) uploaded successfully.`)
  )
})

const deleteDataset = asyncHandler(async (req, res) => {
  const { projectId, modelId, datasetId } = req.params;

  const project = await Project.findById(projectId);
  if (!project) {
    throw new ApiError(404, "Project not found.");
  }

  const model = await Model.findOne({ _id: modelId, projectId });
  if (!model) {
    throw new ApiError(404, "Model not found or does not belong to this project.");
  }

  const datasetInfo = await Dataset.findByIdAndDelete(datasetId);

  return res.status(201).json(
    new ApiResponse(200, datasetInfo, "Dataset file deleted successfully.")
  )
})

export {
  getDatasets,
  createDataset,
  deleteDataset
}