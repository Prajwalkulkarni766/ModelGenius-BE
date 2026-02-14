import { Project } from "../models/project.model.js"
import { ApiError } from "../utils/ApiError.js"
import { ApiResponse } from "../utils/ApiResponse.js"
import { asyncHandler } from "../utils/asyncHandler.js"
import { Model } from "../models/model.model.js"
import { Dataset } from '../models/dataset.model.js';
import fs from 'fs/promises';
import path from 'path';
import axios from "axios";

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
    projectId
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

const trainDryRunModel = asyncHandler(async (req, res) => {
  const { projectId, modelId } = req.params;
  const {
    algorithm,
    handlingMissingValueStrategy,
    encodingCategoricalMethod,
    normalizationTechnique,
    targetColumn
  } = req.body;

  const model = await Model.findOne({ _id: modelId, projectId });
  if (!model) {
    throw new ApiError(404, "Model not found");
  }

  const dataset = await Dataset.findById(model.datasetId);
  if (!dataset) {
    throw new ApiError(404, "Dataset not found for this model");
  }

  const payload = {
    dataset_path: dataset.datasetFilePath,
    cleaning_strategy: handlingMissingValueStrategy || model.handlingMissingValueStrategy,
    encoding_method: encodingCategoricalMethod || model.encodingCategoricalMethod,
    normalization_technique: normalizationTechnique || model.normalizationTechnique,
    algorithm: algorithm || model.algorithm,
    target_column: targetColumn || model.targetColumn
  };

  console.log("dry run payload", payload)

  const response = await axios.post(
    process.env.PYTHON_MICROSERVICE,
    payload
  );

  console.log("dry run response", response.data)

  const {
    problem_type,
    accuracy,
    precision,
    recall,
    f1_score,
    mse,
    rmse,
    r2_score,
    model_path
  } = response.data;

  if (problem_type === "regression") {
    return res.status(200).json(
      new ApiResponse(200, { mse, rmse, r2_score }, "Model dry run completed successfully")
    );
  } else {
    return res.status(200).json(
      new ApiResponse(200, { accuracy, precision, recall, f1_score }, "Model dry run completed successfully")
    );
  }
});

const trainModel = asyncHandler(async (req, res) => {
  const { projectId, modelId } = req.params;

  const model = await Model.findOne({ _id: modelId, projectId });
  if (!model) {
    throw new ApiError(404, "Model not found");
  }

  const dataset = await Dataset.findById(model.datasetId);
  if (!dataset) {
    throw new ApiError(404, "Dataset not found for this model");
  }

  const payload = {
    dataset_path: dataset.datasetFilePath,
    cleaning_strategy: model.handlingMissingValueStrategy,
    encoding_method: model.encodingCategoricalMethod,
    normalization_technique: model.normalizationTechnique,
    algorithm: model.algorithm,
    target_column: model.targetColumn
  };

  console.log("payload", payload)

  const response = await axios.post(
    process.env.PYTHON_MICROSERVICE,
    payload
  );

  console.log("response", response.data)

  const {
    problem_type,
    accuracy,
    precision,
    recall,
    f1_score,
    mse,
    rmse,
    r2_score,
    model_path
  } = response.data;

  model.modelPath = model_path;

  if (problem_type === "regression") {
    model.metrics = { mse, rmse, r2_score };
  } else {
    model.metrics = { accuracy, precision, recall, f1_score };
  }

  await model.save();

  if (problem_type === "regression") {
    return res.status(200).json(
      new ApiResponse(200, { mse, rmse, r2_score, modelPath: model_path }, "Model trained successfully")
    );
  } else {
    return res.status(200).json(
      new ApiResponse(200, { accuracy, precision, recall, f1_score, modelPath: model_path }, "Model trained successfully")
    );
  }
});

const exportModel = asyncHandler(async (req, res) => {
  const { projectId, modelId } = req.params;

  const project = await Project.findById(projectId);
  if (!project) {
    throw new ApiError(404, "Project not found.");
  }

  const model = await Model.findById(modelId);
  if (!model) {
    throw new ApiError(404, "Model not found.");
  }

  if (model.projectId.toString() !== projectId) {
    throw new ApiError(403, "Model does not belong to this project.");
  }

  if (!model.modelPath) {
    throw new ApiError(404, "Model file path not found. Please train the model first.");
  }

  const filePath = path.resolve(model.modelPath);

  try {
    await fs.access(filePath);
  } catch (error) {
    throw new ApiError(404, "Model file not found on server.");
  }

  return res.download(filePath, (err) => {
    if (err) {
      console.error("File download error:", err);
      if (!res.headersSent) {
        res.status(500).send("Could not download file.");
      }
    }
  });
});

const exportModelCode = asyncHandler(async (req, res) => {
  const { projectId, modelId } = req.params;

  const project = await Project.findById(projectId);
  if (!project) {
    throw new ApiError(404, "Project not found.");
  }

  const model = await Model.findById(modelId);
  if (!model) {
    throw new ApiError(404, "Model not found.");
  }

  if (model.projectId.toString() !== projectId) {
    throw new ApiError(403, "Model does not belong to this project.");
  }

  const payload = {
    target_column: model.targetColumn || "",
    algorithm: model.algorithm || "random_forest",
    cleaning_strategy: model.handlingMissingValueStrategy || "drop_rows",
    encoding_method: model.encodingCategoricalMethod || "one_hot",
    normalization_technique: model.normalizationTechnique || "zscore"
  };

  try {
    const response = await axios.post(
      `${process.env.PYTHON_MICROSERVICE?.replace('/train', '')}/generate-code`,
      payload,
      { responseType: 'text' }
    );

    const code = response.data;
    const fileName = `${model.modelName || 'model'}_training_code.py`;

    res.setHeader('Content-Type', 'text/x-python');
    res.setHeader('Content-Disposition', `attachment; filename="${fileName}"`);
    return res.send(code);
  } catch (error) {
    console.error("Code generation error:", error.message);
    throw new ApiError(500, "Failed to generate Python code.");
  }
});

const aiChat = asyncHandler(async (req, res) => {
  const { projectId, modelId } = req.params;
  const { message, chatHistory } = req.body;

  if (!message) {
    throw new ApiError(400, "Message is required");
  }

  const model = await Model.findOne({ _id: modelId, projectId });
  if (!model) {
    throw new ApiError(404, "Model not found");
  }

  const dataset = await Dataset.findById(model.datasetId);

  const systemPrompt = `You are an AI assistant for ModelGenius, a machine learning platform.
You are helping the user with their model "${model.modelName || 'Unnamed Model'}".

Model Configuration:
- Algorithm: ${model.algorithm || 'Not set'}
- Target Column: ${model.targetColumn || 'Not set'}
- Encoding Method: ${model.encodingCategoricalMethod || 'Not set'}
- Normalization: ${model.normalizationTechnique || 'Not set'}
- Missing Value Strategy: ${model.handlingMissingValueStrategy || 'Not set'}
${model.metrics ? `Performance Metrics: ${JSON.stringify(model.metrics)}` : 'Model has not been trained yet.'}
${model.modelPath ? 'Model is trained and saved.' : 'Model has not been trained yet.'}
${dataset ? `Dataset: ${dataset.datasetName || 'Unknown file'}` : 'No dataset associated with this model.'}

Help the user understand their model, suggest improvements, explain metrics, 
and recommend better preprocessing or algorithm choices based on the context above.
Keep responses concise and actionable.`;

  const messages = [
    { role: 'system', content: systemPrompt },
    ...(chatHistory || []),
    { role: 'user', content: message }
  ];

  try {
    const response = await axios.post(
      'https://openrouter.ai/api/v1/chat/completions',
      {
        model: 'arcee-ai/trinity-large-preview:free',
        messages: messages
      },
      {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${process.env.OPENROUTER_API_KEY}`
        }
      }
    );

    const reply = response.data.choices[0].message.content;

    return res.status(200).json(
      new ApiResponse(200, { reply }, "Chat response generated successfully")
    );
  } catch (error) {
    console.error("AI Chat error:", error.message);
    throw new ApiError(500, "Failed to get AI response");
  }
});

export {
  getModels,
  getModel,
  createModel,
  updateModel,
  deleteModel,
  trainModel,
  trainDryRunModel,
  exportModel,
  exportModelCode,
  aiChat
}
