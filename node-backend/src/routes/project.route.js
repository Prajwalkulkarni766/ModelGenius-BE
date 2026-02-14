import { Router } from 'express';
import {
  createProject,
  getUserProjects,
  updateProject,
  deleteProject,
  getUserLatestProjects,
  getUserProject
} from "../controllers/project.controller.js"
import { verifyJWT } from "../middlewares/auth.middleware.js"
import { uploadProjectImage } from "../middlewares/multer.middleware.js"

const router = Router();
router.use(verifyJWT);

// Project route
router.route("/").post(uploadProjectImage.fields([
  {
    name: "projectFile",
    maxCount: 1,
  },
]), createProject);
router.route("/projects").get(getUserProjects);
router.route("/latest").get(getUserLatestProjects);
router.route("/:projectId").get(getUserProject);
router.route("/:projectId").patch(updateProject);
router.route("/:projectId").delete(deleteProject);

export default router