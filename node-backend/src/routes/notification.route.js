import { Router } from 'express';
import {
  createNotification,
  getUserNotifications,
} from "../controllers/notification.controller.js"
import { verifyJWT } from "../middlewares/auth.middleware.js"

const router = Router();
router.use(verifyJWT);

router.route("/").post(createNotification);
router.route("/").get(getUserNotifications);

export default router