import { Router } from 'express';
import {
  getDatasets,
  createDataset,
  deleteDataset,
  getDatasetColumns,
  getDatasetPreview
} from "../controllers/dataset.controller.js"
import { verifyJWT } from "../middlewares/auth.middleware.js"
import { uploadDataset } from "../middlewares/multer.middleware.js"

const router = Router();
router.use(verifyJWT);

router.route("/:projectId/datasets").post(uploadDataset.fields([
  {
    name: "datasetFile",
  },
]), createDataset);
router.route("/:projectId/datasets").get(getDatasets);
router.route("/:projectId/datasets/:datasetId").delete(deleteDataset);
router.route("/:projectId/datasets/:datasetId/columns").get(getDatasetColumns);
router.route("/:projectId/datasets/:datasetId/preview").get(getDatasetPreview);

export default router