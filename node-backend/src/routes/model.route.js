import { Router } from 'express';
import {
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
} from "../controllers/model.controller.js"
import { verifyJWT } from "../middlewares/auth.middleware.js"

const router = Router();
router.use(verifyJWT);

// Model route
router.route("/:projectId/models").get(getModels);
router.route("/:projectId/models").post(createModel);
router.route("/:projectId/models/:modelId/train").post(trainModel);
router.route("/:projectId/models/:modelId/train-dry-run").post(trainDryRunModel);
router.route("/:projectId/models/:modelId/export").get(exportModel);
router.route("/:projectId/models/:modelId/export-code").get(exportModelCode);
router.route("/:projectId/models/:modelId/ai-chat").post(aiChat);
router.route("/:projectId/models/:modelId").get(getModel);
router.route("/:projectId/models/:modelId").patch(updateModel);
router.route("/:projectId/models/:modelId").delete(deleteModel);

export default router