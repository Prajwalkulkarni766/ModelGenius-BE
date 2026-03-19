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

const router = Router();
router.use(verifyJWT);

// Project route
router.route("/")
  .get(getUserProjects)
  .post(createProject);
router.route("/latest").get(getUserLatestProjects);
router.route("/:projectId")
  .get(getUserProject)
  .patch(updateProject)
  .delete(deleteProject);

export default router