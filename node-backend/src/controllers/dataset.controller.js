import { Dataset } from "../models/dataset.model.js"
import { Project } from "../models/project.model.js"
import { ApiError } from "../utils/ApiError.js"
import { ApiResponse } from "../utils/ApiResponse.js"
import { asyncHandler } from "../utils/asyncHandler.js"
import fs from "fs/promises"
import fsStandard from "fs"
import path from "path"
import csv from "csv-parser"
import { extractCsvColumns } from "../utils/extractCsvColumns.js";

const getDatasets = asyncHandler(async (req, res) => {
  const { projectId } = req.params;

  const project = await Project.findById(projectId);
  if (!project) {
    throw new ApiError(404, "Project not found.");
  }

  const dataset = await Dataset.find({ projectId }).select("-datasetFilePath -projectId");

  return res.status(200).json(
    new ApiResponse(200, dataset, "Dataset fetched Successfully")
  )
})

const createDataset = asyncHandler(async (req, res) => {

  const { projectId } = req.params;

  const project = await Project.findById(projectId);
  if (!project) {
    throw new ApiError(404, "Project not found.");
  }

  const uploadedFiles = req.files?.datasetFile;

  if (!uploadedFiles || uploadedFiles.length === 0) {
    throw new ApiError(400, "No dataset files uploaded.");
  }

  // Create a dataset entry for each uploaded file
  const createdDatasets = await Promise.all(
    uploadedFiles.map(async (file) => {
      const columns = await extractCsvColumns(file.path);

      return Dataset.create({
        datasetFilePath: file.path,
        originalFileName: file.originalname,
        fileSize: file.size,
        columns,
        projectId
      });
    })
  );

  return res.status(201).json(
    new ApiResponse(200, createdDatasets, `${createdDatasets.length} dataset file(s) uploaded successfully.`)
  )
})

const deleteDataset = asyncHandler(async (req, res) => {
  const { projectId, datasetId } = req.params;

  const project = await Project.findById(projectId);
  if (!project) {
    throw new ApiError(404, "Project not found.");
  }

  const dataset = await Dataset.findById(datasetId);
  if (!dataset) {
    throw new ApiError(404, "Dataset not found.");
  }

  if (dataset.projectId.toString() !== projectId) {
    throw new ApiError(403, "Dataset does not belong to this project.");
  }

  if (dataset.datasetFilePath) {
    const filePath = path.resolve(dataset.datasetFilePath);

    try {
      await fs.unlink(filePath);
    } catch (err) {
      console.error("File delete error:", err);
    }
  }

  await Dataset.findByIdAndDelete(datasetId);

  return res.status(200).json(
    new ApiResponse(200, dataset, "Dataset file deleted successfully.")
  )
})

const getDatasetColumns = asyncHandler(async (req, res) => {
  const { projectId, datasetId } = req.params;

  const project = await Project.findById(projectId);
  if (!project) {
    throw new ApiError(404, "Project not found.");
  }

  const dataset = await Dataset.findById(datasetId).select("columns projectId");
  if (!dataset) {
    throw new ApiError(404, "Dataset not found.");
  }

  if (dataset.projectId.toString() !== projectId) {
    throw new ApiError(403, "Dataset does not belong to this project.");
  }

  return res.status(200).json(
    new ApiResponse(200, dataset.columns, "Dataset columns fetched successfully")
  );
});


export {
  getDatasets,
  createDataset,
  deleteDataset,
  getDatasetColumns,
  getDatasetPreview
}

const getDatasetPreview = asyncHandler(async (req, res) => {
  const { projectId, datasetId } = req.params;

  const project = await Project.findById(projectId);
  if (!project) {
    throw new ApiError(404, "Project not found.");
  }

  const dataset = await Dataset.findById(datasetId);
  if (!dataset) {
    throw new ApiError(404, "Dataset not found.");
  }

  if (dataset.projectId.toString() !== projectId) {
    throw new ApiError(403, "Dataset does not belong to this project.");
  }

  if (!dataset.datasetFilePath) {
    throw new ApiError(404, "Dataset file path not found.");
  }

  const filePath = path.resolve(dataset.datasetFilePath);
  const results = [];
  const limit = 20;

  try {
    await new Promise((resolve, reject) => {
      const stream = fsStandard.createReadStream(filePath)
        .pipe(csv());

      stream.on("data", (data) => {
        if (results.length < limit) {
          results.push(data);
        } else {
          stream.destroy(); // Stop reading after limit
          resolve();
        }
      })
        .on("end", () => {
          resolve();
        })
        .on("error", (error) => {
          reject(error);
        });

      // Handle close event if destroyed manually
      stream.on("close", () => resolve());
    });
  } catch (error) {
    // If error is just stream destroy, ignore or handle
    // Actually csv-parser might emit error on destroy in some versions/contexts but usually fine.
    // If real error, throw
    console.error("Error reading CSV preview:", error);
    throw new ApiError(500, "Failed to read dataset preview.");
  }

  return res.status(200).json(
    new ApiResponse(200, results, "Dataset preview fetched successfully")
  );
});