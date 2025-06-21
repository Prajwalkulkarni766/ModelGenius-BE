import { Router } from 'express';
import {
  getDatasets,
  createDataset,
  deleteDataset
} from "../controllers/dataset.controller.js"
import { verifyJWT } from "../middlewares/auth.middleware.js"
import { uploadDataset } from "../middlewares/multer.middleware.js"

const router = Router();
router.use(verifyJWT);

router.route("/:projectId/models/:modelId/datasets").get(getDatasets);
router.route("/:projectId/models/:modelId/datasets").post(uploadDataset.fields([
  {
    name: "datasetFile",
  },
]), createDataset);
router.route("/:projectId/models/:modelId/datasets/:datasetId").delete(deleteDataset);

export default router