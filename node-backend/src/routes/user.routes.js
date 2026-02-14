import { Router } from "express";
import {
    loginUser,
    logoutUser,
    registerUser,
    refreshAccessToken,
    changeCurrentPassword,
    getCurrentUser,
    updateUserAvatar,
    getUserChannelProfile,
    getWatchHistory,
    updateAccountDetails,
    deleteAccount,
    getProfile
} from "../controllers/user.controller.js";
import { uploadAvatarImage } from "../middlewares/multer.middleware.js"
import { verifyJWT } from "../middlewares/auth.middleware.js";


const router = Router()

router.route("/register").post(
    uploadAvatarImage.fields([
        {
            name: "avatar",
            maxCount: 1
        },
    ]),
    registerUser
)

router.route("/login").post(loginUser)

//secured routes
router.route("/logout").post(verifyJWT, logoutUser)
router.route("/refresh-token").post(refreshAccessToken)
router.route("/change-password").post(verifyJWT, changeCurrentPassword)
router.route("/current-user").get(verifyJWT, getCurrentUser)
router.route("/update-account").patch(verifyJWT, updateAccountDetails)
router.route("/delete-account").delete(verifyJWT, deleteAccount)

router.route("/avatar").patch(verifyJWT, uploadAvatarImage.single("avatar"), updateUserAvatar)
router.route("/profile").get(verifyJWT, getProfile)
router.route("/c/:username").get(verifyJWT, getUserChannelProfile)
router.route("/history").get(verifyJWT, getWatchHistory)

export default router