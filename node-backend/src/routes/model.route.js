import { Router } from 'express';
import {
  getModels,
  getModel,
  createModel,
  updateModel,
  deleteModel
} from "../controllers/model.controller.js"
import { verifyJWT } from "../middlewares/auth.middleware.js"

const router = Router();
router.use(verifyJWT);

// Model route
router.route("/:projectId/models").get(getModels);
router.route("/:projectId/models").post(createModel);
router.route("/:projectId/models/:modelId").get(getModel);
router.route("/:projectId/models/:modelId").patch(updateModel);
router.route("/:projectId/models/:modelId").delete(deleteModel);

// GET    /projects/:projectId/models/:modelId/datasets
// POST   /projects/:projectId/models/:modelId/datasets
// GET    /projects/:projectId/models/:modelId/datasets/:datasetId
// PUT    /projects/:projectId/models/:modelId/datasets/:datasetId
// DELETE /projects/:projectId/models/:modelId/datasets/:datasetId

  


// GET    /projects/:projectId/datasets             # List datasets in a project
// POST   /projects/:projectId/datasets             # Create a new dataset in the project
// GET    /projects/:projectId/datasets/:datasetId  # Get a specific dataset
// PUT    /projects/:projectId/datasets/:datasetId  # Update a dataset
// DELETE /projects/:projectId/datasets/:datasetId  # Delete a dataset


export default router