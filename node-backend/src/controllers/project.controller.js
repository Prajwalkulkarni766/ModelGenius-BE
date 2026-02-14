import { Project } from "../models/project.model.js"
import { Model } from "../models/model.model.js"
import { Dataset } from "../models/dataset.model.js"
import { ApiError } from "../utils/ApiError.js"
import { ApiResponse } from "../utils/ApiResponse.js"
import { asyncHandler } from "../utils/asyncHandler.js"
import path from "path"

const createProject = asyncHandler(async (req, res) => {

  let filePath = null;

  if (req.files && req.files.projectFile && req.files.projectFile.length > 0) {
    filePath = `public/images/${path.basename(req.files.projectFile[0].path)}`;
  }

  const { projectTitle, projectDescription } = req.body;
  const { _id } = req.user;

  if (!projectTitle || !projectDescription) {
    throw new ApiError(400, "All fields are required")
  }

  const project = await Project.create({
    projectTitle,
    projectDescription,
    userId: _id,
    projectFile: filePath
  });

  return res.status(201).json(
    new ApiResponse(200, project, "Project created Successfully")
  )
})

const getUserProjects = asyncHandler(async (req, res) => {
  const { _id } = req.user;

  if (!_id) {
    return res.status(400).json(new ApiError(400, "User ID is required"));
  }

  const projects = await Project.find({ userId: _id }).sort({
    createdAt: -1
  })

  return res.status(200).json(
    new ApiResponse(200, projects, "Projects fetched Successfully")
  )
})

const getUserProject = asyncHandler(async (req, res) => {
  const { projectId } = req.params;

  const projectDetails = await Project.findById(projectId);

  const modelsRealtedToThisProject = await Model.find({ projectId }).select("modelName createdAt")

  const datasetRelatedToThisProject = await Dataset.find({ projectId })

  // TODO: send models and datasets with the project data
  // under model details - send modelName, mlModelName and updatedAt

  return res.status(200).json(
    new ApiResponse(200, { projectDetails, modelsRealtedToThisProject, datasetRelatedToThisProject }, "Project detail fetched Successfully")
  )
})

const getUserLatestProjects = asyncHandler(async (req, res) => {

  const { _id } = req.user;

  if (!_id) {
    return res.status(400).json(new ApiError(400, "User ID is required"));
  }

  const projects = await Project.find({ userId: _id }).sort({
    createdAt: -1
  }).limit(2)

  return res.status(200).json(
    new ApiResponse(200, projects, "Projects fetched Successfully")
  )
})

const updateProject = asyncHandler(async (req, res) => {
  const { title, description } = req.body
  const { projectId } = params

  if (!projectId || !title || !description) {
    throw new ApiError(400, "All fields are required")
  }

  const project = await Project.findByIdAndUpdate(
    projectId,
    {
      $set: {
        title,
        description
      }
    },
    { new: true }
  );

  return res
    .status(200)
    .json(new ApiResponse(200, project, "Project details updated successfully"))
})

const deleteProject = asyncHandler(async (req, res) => {
  const { projectId } = req.params;

  if (!projectId) {
    throw new ApiError(400, "All fields are required")
  }

  const project = await Project.findByIdAndDelete(projectId);

  return res
    .status(204)
    .json(new ApiResponse(204, project, "Project deleted successfully"))
})

export {
  createProject,
  getUserProjects,
  updateProject,
  deleteProject,
  getUserLatestProjects,
  getUserProject
}
