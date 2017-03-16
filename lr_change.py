import cv2
import numpy as np
import os
from random import shuffle

# from BK.MeshNetLayer import *
from Lib3D.MeshModel import *
from Lib3D.Rasterization import  *
from Lib3D.Util2D import *
from Lib3D.Util3D import *
from Lib3D.LocalFeatures import *
from Lib3D.ParamTransfer import *
from keras import optimizers
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def final_pred(y_true, y_pred):
    # y_cont=np.concatenate(y_pred,axis=1)
    return y_pred
batch_size=8

TN = TextNet('./MatBS/shape_0.obj', imgW=256,bsize=batch_size)
TN_HD = TextNet('./MatBS/shape_0.obj', imgW=256,Pj_imgW=256)
TrainPath = '/temp/xiaoshengtao/data/2D_Images/Croped256/Script/KBKC4_train.txt'
TestPath = '/temp/xiaoshengtao/data/2D_Images/300W/300WTest.txt'

TrainPath = '/home/shengtao/Data/2D_Images/Croped256/Script/KBKC4_train.txt'  # iter 1
# TrainPath = '/home/shengtao/Data/2D_Images/Croped256/Script/MX3H_train.txt'    # iter 2
TestPath = '/home/shengtao/Data/2D_Images/300W/CropBB15/300WBB15.txt'

FTr = open(TrainPath,'r')
DataTr = FTr.readlines()
FTe = open(TestPath,'r')
DataTe = FTe.readlines()
# TrData,TrLabel=load_train_data(DataTr,0,5,5)


if os.path.exists('./STD.txt'):
    STD = np.loadtxt('./STD.txt')
    STD[57:] = 1
    Mean = np.loadtxt('./Mean.txt')
    Mean[57:]= 0
else:
    shuffle(DataTr)
    DataLabels = np.zeros([len(DataTr) / 5, 104])
    for i in range(len(DataTr) / 5):
        # i=0
        crtLabel = DataTr[i]
        X = crtLabel.rstrip(' \n').split(' ')[137:]
        DataLabels[i, :] = np.asarray(X).astype(np.float32)
    Mean = np.mean(DataLabels,axis=0)
    STD = np.std(DataLabels, axis=0)
    STD[57:] = 1
    Mean[57:] = 0
    np.savetxt('./STD.txt',STD)
    np.savetxt('./Mean.txt', STD)


Vpath = './MatBS/shape_0.obj'
VFid = open(Vpath, 'r')
VFlines = VFid.readlines()


TrNum = len(DataTr)
MaxIters = TrNum/batch_size

modelInit = MeshNetV2(weights_path='./FD/MeshNet_V3_rl.h5',imgW=128,ZK3D=TN)

modelTextPCA = TextGenPCANet(weights_path='./FD/TextNetPCA_V3_rl.h5',imgW=136)
modelRefinerPTS = MeshNetAB_Update_PTS(weights_path='./FDABPTS/RefNetPTS_v1_80k.h5',imgW=128,input_channels=6)


# modelRefinerPTS_R2 = MeshNetAB_Update_PTS_Pool(weights_path='./FDABPTS_P4/RefNetPTS_RC_v2_86000.h5',imgW=128,input_channels=6)
# modelSLF_LocalFeat.save(SavePath + './R2IDFixed/SFL_LocalFeat_v1_%d.h5' % iter)
# modelSLF_LocalDecoder.save(SavePath + './R2IDFixed/SFL_LocalDecoder_v1_%d.h5' % iter)
# SFL_GlobalNet.save(SavePath + './R2IDFixed/SFL_GlobalDecoder_v1_%d.h5' % iter)

SavePath='./R2ID_GRLF_FitEXP/'
KSIZE=7
# modelRefinerPTS_R2 = MeshNetAB_Update_PTS_FC(imgW=128,input_channels=3)
modelSLF_LocalFeat =SFL_Encoder_P4(weights_path=SavePath+'SFL_LocalFeat_v1_75000.h5',ksize=KSIZE,input_channels=6)
print modelSLF_LocalFeat.summary()
modelSLF_LocalDecoder = SFL_LocalDecoder(weights_path=SavePath+'SFL_LocalDecoder_v1_75000.h5',output_dim=2)
print modelSLF_LocalDecoder.summary()
SLF_LocalNet=SFL_LocalEncoderDecoder(modelSLF_LocalFeat,modelSLF_LocalDecoder,KSize=KSIZE,input_channels=6)
print SLF_LocalNet.summary()
SFL_GlobalNet=SFL_GlobalDecoder(weights_path=SavePath+'SFL_GlobalDecoder_v1_75000.h5',input_dim=128)


modelSLF_LocalFeat2 =SFL_Encoder(weights_path=SavePath+'SFL_LocalFeat2_v1_20000.h5', ksize=KSIZE,input_channels=6)
print modelSLF_LocalFeat2.summary()
modelSLF_LocalDecoder2 = SFL_LocalDecoder(weights_path=SavePath+'SFL_LocalDecoder2_v1_20000.h5',output_dim=2)
print modelSLF_LocalDecoder2.summary()
SLF_LocalNet2=SFL_LocalEncoderDecoder(modelSLF_LocalFeat2,modelSLF_LocalDecoder2,KSize=KSIZE,input_channels=6)
print SLF_LocalNet2.summary()
SFL_GlobalNet2=SFL_GlobalDecoder(weights_path=SavePath+'SFL_GlobalDecoder2_v1_20000.h5',input_dim=128)



PCA = sio.loadmat('./TextureDB/ABMapPCA2.mat')
Psize = 150
PCA_ST =  PCA['St'][:Psize,:Psize]
PCA_VT = PCA['Vt'][:,:Psize]
Rt = np.dot(np.sqrt(PCA_ST),PCA_VT.transpose())
invRt = np.linalg.inv(np.dot(Rt,Rt.transpose()))
PCA_Mean = PCA['ABMean']
PCA_MeanRep = PCA_Mean[np.newaxis,...].repeat(batch_size,0)

def train_epoch(nepoch,SavePath,MaxIters):
    if os.path.exists(SavePath)==False:
        os.mkdir(SavePath)
    for e in range(nepoch):
        # if e>0:
        shuffle(DataTr)
        for iter in range (MaxIters):
            train_start=iter*batch_size
            train_end = (iter+1)*batch_size
            # X_batch, Y_batchO, Z_Names = load_train_data(DataTr,train_start,train_end,batch_size,isTrain=True,isPTS=True)
            X_batch, X_batch_HD, Y_batchO, Z_Names = load_train_data(DataTr, train_start, train_end, batch_size,
                                                                     isTrain=True, isPTS=True,isScale=True, HDSize=256)
            Y_batchGT=Y_batchO[:,136:]
            Y_ptsGT = Y_batchO[:,:136]
            Y_batch = (Y_batchGT - Mean) / STD
            ########################## Initial MeshNet ################################
            PredPYR, PredT, PredS, PredID, PredEXP = modelInit.predict(X_batch)
            Pred3D_Init = np.concatenate([PredPYR, PredT, PredS, PredID, PredEXP], axis=1)*STD+Mean
            predPTS_XYZ = TN.ZK.predictPtswith3D(Pred3D_Init, imgW=128)
            predPTS_Init = predPTS_XYZ[:, :, :2].reshape(batch_size, 136)  # /128-0.5
            Mesh3D_Init= TN.ZK.construct3DSampledMeshBacthEfficient(Pred3D_Init, imgW=128)

            ########################## TextNet ################################
            ABMap_X = TN.extractABbyBatch(Mesh3D_Init, X_batch)
            TextPredPCA = modelTextPCA.predict(ABMap_X)
            RectPCAMapPred = np.dot(TextPredPCA, Rt).reshape([batch_size, 136, 136, 3], order="F") + PCA_MeanRep
            RectPCAMapPred[RectPCAMapPred < 0] = 0
            RectPCAMapPred[RectPCAMapPred > 255] = 255

            ########################### RefineNet ################################
            PJImages=TN.getProjImagebyBatch(Mesh3D_Init, RectPCAMapPred,bsize=batch_size)
            #### Train RefineNet
            RefineInput_Image = np.concatenate([X_batch,PJImages],axis=3)
            pred_Ref1= modelRefinerPTS.predict([RefineInput_Image,ABMap_X,PredPYR,PredT,PredS,PredID,PredEXP,predPTS_Init])
            PredPYR_Ref1,PredT_Ref1,PredS_Ref1,PredID_Ref1,PredEXP_Ref1,PredPTS_Ref1 = pred_Ref1

            PredLabel_Ref1 = np.concatenate([PredPYR_Ref1,PredT_Ref1,PredS_Ref1,PredID_Ref1,PredEXP_Ref1],axis=1) * STD + Mean

            ########################### Image Zoomin and adjust rotation angle ################################

            ############################################# STEP1 ###############################################
            transferedData=transferParamImg(TN,PredLabel_Ref1,PredPTS_Ref1,Y_batchGT[:,0:7],Y_ptsGT,X_batch_HD,Mean,STD,batch_size,dist=0.1,useGT=True,Brate=1.2,imgOrgWidth=256.0,FitEXP=True)
            X_batch_RC, PredLabel_Ref1_RC_ROT, KeyPTS_PJ_1_ROT, predPTS_Ref1_RC, gtCam_RC, gtCam_RC_Norm, gtPTS_RC, predPTS_XYZ_ref1 = transferedData

            Mesh3D_Ref1_RC_Rot = TN.ZK.construct3DSampledMeshBacthEfficient(PredLabel_Ref1_RC_ROT,imgW=128)  ## final 3d used for projection texture
            PJImages_Ref1_RCRot = TN.getProjImagebyBatch(Mesh3D_Ref1_RC_Rot, RectPCAMapPred, bsize=batch_size)
            PredLabel_Ref1_RC_ROT_Norm = (PredLabel_Ref1_RC_ROT.copy() - Mean) / STD  ## normalize for regression

            RefineInput_Image_R2 = np.concatenate([X_batch_RC, PJImages_Ref1_RCRot.astype(int)], axis=3).astype(np.float32)
            SLF = Spatial_Local_Feature_V2(RefineInput_Image_R2, KeyPTS_PJ_1_ROT, ksize=KSIZE)
            SLF_NetInput = SLF.reshape([batch_size * 68, KSIZE * 2 + 1, KSIZE * 2 + 1, 6])

            ###### train local Network 1
            Residual = gtPTS_RC * 128 + 64 - KeyPTS_PJ_1_ROT
            Residual_BL_2 = Residual.reshape([batch_size * 68, 2])
            # resLoss1, trash, predRes = SLF_LocalNet.predict(SLF_NetInput)
            # if iter<5000:
                # print resLoss1
            ##### train global refiner
            SFL_LocalFCFeatures = modelSLF_LocalFeat.predict(SLF_NetInput)
            SFL_LocalFeatureBatch = SFL_LocalFCFeatures.reshape([batch_size, 68 * 128])
            updateID, updateEXP,updatePTS= SFL_GlobalNet.predict(SFL_LocalFeatureBatch)


            PredID_Ref2 = PredLabel_Ref1_RC_ROT_Norm[:,7:57] + updateID
            PredEXP_Ref2 = PredLabel_Ref1_RC_ROT_Norm[:,57:] + updateEXP
            PredPTS_Ref2 = KeyPTS_PJ_1_ROT + updatePTS
            PredLabel_Ref2 = np.concatenate([PredLabel_Ref1_RC_ROT_Norm[:, :7], PredID_Ref2, PredEXP_Ref2],
                                            axis=1) * STD + Mean

            transferedData2 = transferParamImg(TN, PredLabel_Ref2, PredPTS_Ref2, gtCam_RC[:, 0:7], gtPTS_RC, X_batch_RC,
                                              Mean, STD, batch_size, dist=0.1, useGT=True, Brate=1.2, imgOrgWidth=256.0,
                                              FitEXP=True,imageRot=False)
            X_batch_RC2, PredLabel_Ref2_RC_ROT, KeyPTS_PJ_2_ROT, predPTS_Ref2_RC, trash1, trash2, trash3, predPTS_XYZ_ref2 = transferedData2

            ####################################################################################################################################
            ################################# STEP2#############################################################################################
            Mesh3D_Ref2_RC_Rot = TN.ZK.construct3DSampledMeshBacthEfficient(PredLabel_Ref2_RC_ROT,
                                                                            imgW=128)  ## final 3d used for projection texture
            PJImages_Ref2_RCRot = TN.getProjImagebyBatch(Mesh3D_Ref2_RC_Rot, RectPCAMapPred, bsize=batch_size)
            PredLabel_Ref2_RC_ROT_Norm = (PredLabel_Ref2_RC_ROT.copy() - Mean) / STD  ## normalize for regression

            RefineInput_Image_R2 = np.concatenate([X_batch_RC, PJImages_Ref2_RCRot.astype(int)], axis=3).astype(
                np.float32)

            SLF2 = Spatial_Local_Feature_V2(RefineInput_Image_R2, KeyPTS_PJ_2_ROT, ksize=KSIZE)
            SLF_NetInput2 = SLF2.reshape([batch_size * 68, KSIZE * 2 + 1, KSIZE * 2 + 1, 6])

            ###### train local Network 1
            Residual2 = gtPTS_RC * 128 + 64 - KeyPTS_PJ_2_ROT
            Residual2_BL_2 = Residual2.reshape([batch_size * 68, 2])
            resLoss2, trash2, predRes2 = SLF_LocalNet2.train_on_batch(SLF_NetInput2, Residual2_BL_2)


            ##### train global refiner
            SFL_LocalFCFeatures2 = modelSLF_LocalFeat2.predict(SLF_NetInput2)
            SFL_LocalFeatureBatch2 = SFL_LocalFCFeatures2.reshape([batch_size, 68 * 128])
            SFL_Trainout2 = SFL_GlobalNet2.train_on_batch(SFL_LocalFeatureBatch2, [Y_batch[:, 7:57] - PredLabel_Ref2_RC_ROT_Norm[:,7:57],
                                                                                Y_batch[:, 57:] - PredLabel_Ref2_RC_ROT_Norm[:,57:],
                                                                                gtPTS_RC * 128 + 64 - KeyPTS_PJ_2_ROT])
            LossTotal,LossID, LossExp, LossPTS,tras, updateID2, tras, updateEXP2, tras, updatePTS2 = SFL_Trainout2

            PredID_Ref3 = PredLabel_Ref2_RC_ROT_Norm[:,7:57] + updateID2
            PredEXP_Ref3 = PredLabel_Ref2_RC_ROT_Norm[:,57:] + updateEXP2
            PredPTS_Ref3 = KeyPTS_PJ_2_ROT + updatePTS2
            PredLabel_Ref3 = np.concatenate([PredLabel_Ref2_RC_ROT_Norm[:, :7], PredID_Ref3, PredEXP_Ref3], axis=1) * STD + Mean

            ##############################################################################################################
            if iter%50==0:
                KK=0
                KeyPTS_PJ_2 = np.reshape(TN.ZK.predictPtswith3D(PredLabel_Ref2, imgW=128)[:,:,:2], [batch_size,68, 2])
                KeyPTS_PJ_3 = np.reshape(TN.ZK.predictPtswith3D(PredLabel_Ref3, imgW=128)[:, :, :2],
                                         [batch_size, 68, 2])

                transferedData3 = transferParamImg(TN, PredLabel_Ref3, PredPTS_Ref3, gtCam_RC[:, 0:7], gtPTS_RC,
                                                   X_batch_RC,
                                                   Mean, STD, batch_size, dist=0.1, useGT=True, Brate=1.2,
                                                   imgOrgWidth=256.0,
                                                   FitEXP=True, imageRot=False)
                X_batch_RC3, PredLabel_Ref3_RC_ROT, KeyPTS_PJ_3_ROT, predPTS_Ref2_RC, trash1, trash2, trash3, predPTS_XYZ_ref2 = transferedData3


                E3D_Step1=0
                E3D_Step1_RV=0
                E3D_Step2=0
                E3D_Step2_RV =0
                E3D_Step3 = 0
                E3D_Step3_RV = 0
                E2D_Step3=0;
                E2D_Step2=0;
                E2D_Step1=0
                gtPTS_RC_rhp=gtPTS_RC.reshape([batch_size,68, 2])*128+64
                PredPTS_Ref3_rhp = PredPTS_Ref3.reshape([batch_size, 68, 2])
                PredPTS_Ref2_rhp=PredPTS_Ref2.reshape([batch_size,68, 2])
                predPTS_Ref1_RC_rhp = predPTS_Ref1_RC.reshape([batch_size, 68, 2])*128+64

                KeyPTS_PJ_1_rhp = predPTS_XYZ_ref1[:batch_size, :, :2].reshape([batch_size, 68, 2])
                KeyPTS_PJ_1_ROT_rhp = KeyPTS_PJ_1_ROT.reshape([batch_size, 68, 2])
                KeyPTS_PJ_2_rhp = KeyPTS_PJ_2.reshape([batch_size, 68, 2])
                KeyPTS_PJ_2_ROT_rhp = KeyPTS_PJ_2_ROT.reshape([batch_size, 68, 2])

                KeyPTS_PJ_3_rhp = KeyPTS_PJ_3.reshape([batch_size, 68, 2])
                KeyPTS_PJ_3_ROT_rhp = KeyPTS_PJ_3_ROT.reshape([batch_size, 68, 2])

                for x in range (batch_size):
                    E2D_Step3 +=evalPTS(gtPTS_RC_rhp[x,...],PredPTS_Ref3_rhp[x,...])
                    E2D_Step2 +=evalPTS(gtPTS_RC_rhp[x,...],PredPTS_Ref2_rhp[x,...])
                    E2D_Step1 += evalPTS(gtPTS_RC_rhp[x,...], predPTS_Ref1_RC_rhp[x,...])

                    E3D_Step1 += evalPTS(gtPTS_RC_rhp[x,...],KeyPTS_PJ_1_rhp[x,...])
                    E3D_Step1_RV += evalPTS(gtPTS_RC_rhp[x,...], KeyPTS_PJ_1_ROT_rhp[x,...])
                    E3D_Step2 += evalPTS(gtPTS_RC_rhp[x,...], KeyPTS_PJ_2_rhp[x,...])
                    E3D_Step2_RV += evalPTS(gtPTS_RC_rhp[x, ...], KeyPTS_PJ_2_ROT_rhp[x, ...])
                    E3D_Step3 += evalPTS(gtPTS_RC_rhp[x, ...], KeyPTS_PJ_3_rhp[x, ...])
                    E3D_Step3_RV += evalPTS(gtPTS_RC_rhp[x, ...], KeyPTS_PJ_3_ROT_rhp[x, ...])

                print 'Train iter ', iter, 'Step2: Total|', LossTotal, ',ID|', LossID, ',EXP|', LossExp, ',PTS|',LossPTS
                print 'Train iter ', iter,'step1: 3d: ', E3D_Step1/batch_size, '3d_rv: ', E3D_Step1_RV/batch_size, '2d: ', E2D_Step1/batch_size
                print 'Train iter ', iter, 'step2: 3d: ', E3D_Step2 / batch_size, '3d_rv: ', E3D_Step2_RV / batch_size, '2d: ', E2D_Step2 / batch_size
                print 'Train iter ', iter, 'step3: 3d: ', E3D_Step3 / batch_size, '3d_rv: ', E3D_Step3_RV / batch_size, '2d: ', E2D_Step3 / batch_size
                #

            if iter%5000==0:
                modelSLF_LocalFeat2.save(SavePath+'/SFL_LocalFeat2_v1_%d.h5'%iter)
                modelSLF_LocalDecoder2.save(SavePath+'/SFL_LocalDecoder2_v1_%d.h5'%iter)
                SFL_GlobalNet2.save(SavePath+'/SFL_GlobalDecoder2_v1_%d.h5'%iter)

                # modelSLF_LocalFeat2.save(SavePath + '/SFL_LocalFeat2_v1_%d.h5' % iter)
                # modelSLF_LocalDecoder2.save(SavePath + '/SFL_LocalDecoder2_v1_%d.h5' % iter)
                # SFL_GlobalNet2.save(SavePath + '/SFL_GlobalDecoder2_v1_%d.h5' % iter)
            if iter%2000 == 0:
                if iter%10000!=0:
                    print '-------------- Testing at iteration %d', iter , '100 samples----------------------'
                    test300W(SavePath=SavePath+'/Test',isSave=False, batch_size=1)
                    print '-----------------------------------------------------------------------'
                if iter%10000==0:
                    print '-------------- Testing at iteration %d', iter , 'full set----------------------'
                    test300W(SavePath=SavePath+'/Test',isSave=False, batch_size=1, TestLens=len(DataTe))
                    print '-----------------------------------------------------------------------'

def test300W(SavePath,isSave=False,batch_size=1,TestLens=100,Srate=50):
    if os.path.exists(SavePath)==False:
        os.mkdir(SavePath)
    PCA_MeanRep = PCA_Mean[np.newaxis, ...].repeat(batch_size, 0)
    KK = 0

    E3D_Step1 = 0
    E3D_Step1_RV = 0
    E3D_Step2 = 0
    E3D_Step2_RV = 0
    E3D_Step3 = 0
    E3D_Step3_RV = 0
    E2D_Step3 = 0;
    E2D_Step2 = 0;
    E2D_Step1 = 0
    shuffle(DataTe)
    for testIdx  in range(TestLens):

        X_batch, X_batch_HD, Y_batchO, Z_Names = load_train_data(DataTe, testIdx, testIdx + batch_size, batch_size,
                                                                 isTrain=False, isScale=True, HDSize=256)
        Y_ptsGT = Y_batchO[:, :136]
        ########################## Initial MeshNet ################################
        PredPYR, PredT, PredS, PredID, PredEXP = modelInit.predict(X_batch)
        Pred3D_Init = np.concatenate([PredPYR, PredT, PredS, PredID, PredEXP], axis=1) * STD + Mean
        predPTS_XYZ = TN.ZK.predictPtswith3D(Pred3D_Init, imgW=128)
        predPTS_Init = predPTS_XYZ[:, :, :2].reshape(batch_size, 136)  # /128-0.5
        Mesh3D_Init = TN.ZK.construct3DSampledMeshBacthEfficient(Pred3D_Init, imgW=128)

        ########################## TextNet ################################
        ABMap_X = TN.extractABbyBatch(Mesh3D_Init, X_batch)
        TextPredPCA = modelTextPCA.predict(ABMap_X)
        RectPCAMapPred = np.dot(TextPredPCA, Rt).reshape([batch_size, 136, 136, 3], order="F") + PCA_MeanRep
        RectPCAMapPred[RectPCAMapPred < 0] = 0
        RectPCAMapPred[RectPCAMapPred > 255] = 255

        ########################### RefineNet ################################
        PJImages_Init = TN.getProjImagebyBatch(Mesh3D_Init, RectPCAMapPred,bsize=batch_size)
        #### Train RefineNet
        RefineInput_Image = np.concatenate([X_batch, PJImages_Init[:batch_size,...]], axis=3)

        pred_Ref1 = modelRefinerPTS.predict(
            [RefineInput_Image, ABMap_X, PredPYR, PredT, PredS, PredID, PredEXP, predPTS_Init])
        PredPYR_Ref1, PredT_Ref1, PredS_Ref1, PredID_Ref1, PredEXP_Ref1, PredPTS_Ref1 = pred_Ref1

        PredLabel_Ref1 = np.concatenate([PredPYR_Ref1, PredT_Ref1, PredS_Ref1, PredID_Ref1, PredEXP_Ref1],
                                        axis=1) * STD + Mean

        ########################### Image Zoomin and adjust rotation angle ################################
        ## step1##############
        transferedData = transferParamImg(TN, PredLabel_Ref1, PredPTS_Ref1, np.zeros([batch_size,7]), Y_ptsGT, X_batch_HD,
                                          Mean, STD, batch_size, dist=0.1, useGT=False, Brate=1.2, imgOrgWidth=256.0,
                                          FitEXP=True,imageRot=True)
        X_batch_RC, PredLabel_Ref1_RC_ROT, KeyPTS_PJ_1_ROT, predPTS_Ref1_RC, gtCam_RC, gtCam_RC_Norm, gtPTS_RC, predPTS_XYZ_ref1 = transferedData

        Mesh3D_Ref1_RC_Rot = TN.ZK.construct3DSampledMeshBacthEfficient(PredLabel_Ref1_RC_ROT,
                                                                        imgW=128)  ## final 3d used for projection texture
        PJImages_Ref1_RCRot = TN.getProjImagebyBatch(Mesh3D_Ref1_RC_Rot, RectPCAMapPred, bsize=batch_size)
        PredLabel_Ref1_RC_ROT_Norm = (PredLabel_Ref1_RC_ROT.copy() - Mean) / STD  ## normalize for regression

        RefineInput_Image_R2 = np.concatenate([X_batch_RC, PJImages_Ref1_RCRot[0:batch_size,...].astype(int)], axis=3).astype(np.float32)
        SLF = Spatial_Local_Feature_V2(RefineInput_Image_R2, KeyPTS_PJ_1_ROT, ksize=KSIZE)
        SLF_NetInput = SLF.reshape([batch_size * 68, KSIZE * 2 + 1, KSIZE * 2 + 1, 6])
        SFL_LocalFCFeatures = modelSLF_LocalFeat.predict(SLF_NetInput)
        SFL_LocalFeatureBatch = SFL_LocalFCFeatures.reshape([batch_size, 68 * 128])
        SFL_Trainout = SFL_GlobalNet.predict(SFL_LocalFeatureBatch)
        updateID, updateEXP, updatePTS = SFL_Trainout

        PredID_Ref2 = PredLabel_Ref1_RC_ROT_Norm[:, 7:57] + updateID
        PredEXP_Ref2 = PredLabel_Ref1_RC_ROT_Norm[:, 57:] + updateEXP
        PredPTS_Ref2 = KeyPTS_PJ_1_ROT + updatePTS
        PredLabel_Ref2 = np.concatenate([PredLabel_Ref1_RC_ROT_Norm[:, :7], PredID_Ref2, PredEXP_Ref2],
                                        axis=1) * STD + Mean

        transferedData2 = transferParamImg(TN, PredLabel_Ref2, PredPTS_Ref2, np.zeros([batch_size,7]), gtPTS_RC, X_batch_RC,
                                          Mean, STD, batch_size, dist=0.1, useGT=False, Brate=1.2, imgOrgWidth=256.0,
                                          FitEXP=True,imageRot=False)
        X_batch_RC, PredLabel_Ref2_RC_ROT, KeyPTS_PJ_2_ROT, predPTS_Ref2_RC, gtCam_RC2, gtCam_RC_Norm2, gtPTS_RC2, predPTS_XYZ_ref2 = transferedData2
        #################### step 2 ################# refinement
        Mesh3D_Ref2_RC_Rot = TN.ZK.construct3DSampledMeshBacthEfficient(PredLabel_Ref2_RC_ROT,
                                                                        imgW=128)  ## final 3d used for projection texture
        PJImages_Ref2_RCRot = TN.getProjImagebyBatch(Mesh3D_Ref2_RC_Rot, RectPCAMapPred, bsize=batch_size)
        PredLabel_Ref2_RC_ROT_Norm = (PredLabel_Ref2_RC_ROT.copy() - Mean) / STD  ## normalize for regression

        RefineInput_Image_R2 = np.concatenate([X_batch_RC, PJImages_Ref2_RCRot[0:batch_size,...].astype(int)], axis=3).astype(np.float32)
        SLF2 = Spatial_Local_Feature_V2(RefineInput_Image_R2, KeyPTS_PJ_2_ROT, ksize=KSIZE)

        SLF_NetInput2 = SLF2.reshape([batch_size * 68, KSIZE * 2 + 1, KSIZE * 2 + 1, 6])
        SFL_LocalFCFeatures2 = modelSLF_LocalFeat2.predict(SLF_NetInput2)
        SFL_LocalFeatureBatch2 = SFL_LocalFCFeatures2.reshape([batch_size, 68 * 128])
        SFL_Trainout2 = SFL_GlobalNet2.predict(SFL_LocalFeatureBatch2)
        updateID2, updateEXP2, updatePTS2 = SFL_Trainout2

        PredID_Ref3 = PredLabel_Ref2_RC_ROT_Norm[:, 7:57] + updateID2
        PredEXP_Ref3 = PredLabel_Ref2_RC_ROT_Norm[:, 57:] + updateEXP2
        PredPTS_Ref3 = KeyPTS_PJ_2_ROT + updatePTS2
        PredLabel_Ref3 = np.concatenate([PredLabel_Ref2_RC_ROT_Norm[:, :7], PredID_Ref3, PredEXP_Ref3],
                                        axis=1) * STD + Mean

        ########### evaluate here

        KeyPTS_PJ_2 = np.reshape(TN.ZK.predictPtswith3D(PredLabel_Ref2, imgW=128)[:, :, :2], [batch_size, 68, 2])
        KeyPTS_PJ_3 = np.reshape(TN.ZK.predictPtswith3D(PredLabel_Ref3, imgW=128)[:, :, :2],
                                 [batch_size, 68, 2])

        transferedData3 = transferParamImg(TN, PredLabel_Ref3, PredPTS_Ref3, gtCam_RC[:, 0:7], gtPTS_RC,
                                           X_batch_RC,
                                           Mean, STD, batch_size, dist=0.1, useGT=True, Brate=1.2,
                                           imgOrgWidth=256.0,
                                           FitEXP=True, imageRot=False)
        X_batch_RC3, PredLabel_Ref3_RC_ROT, KeyPTS_PJ_3_ROT, predPTS_Ref2_RC, trash1, trash2, trash3, predPTS_XYZ_ref2 = transferedData3


        gtPTS_RC_rhp = gtPTS_RC.reshape([batch_size, 68, 2]) * 128 + 64
        PredPTS_Ref3_rhp = PredPTS_Ref3.reshape([batch_size, 68, 2])
        PredPTS_Ref2_rhp = PredPTS_Ref2.reshape([batch_size, 68, 2])
        predPTS_Ref1_RC_rhp = predPTS_Ref1_RC.reshape([batch_size, 68, 2]) * 128 + 64

        KeyPTS_PJ_1_rhp = predPTS_XYZ_ref1[:batch_size, :, :2].reshape([batch_size, 68, 2])
        KeyPTS_PJ_1_ROT_rhp = KeyPTS_PJ_1_ROT.reshape([batch_size, 68, 2])
        KeyPTS_PJ_2_rhp = KeyPTS_PJ_2.reshape([batch_size, 68, 2])
        KeyPTS_PJ_2_ROT_rhp = KeyPTS_PJ_2_ROT.reshape([batch_size, 68, 2])

        KeyPTS_PJ_3_rhp = KeyPTS_PJ_3.reshape([batch_size, 68, 2])
        KeyPTS_PJ_3_ROT_rhp = KeyPTS_PJ_3_ROT.reshape([batch_size, 68, 2])
        #
        x=0
        E2D_Step3 += evalPTS(gtPTS_RC_rhp[x, ...], PredPTS_Ref3_rhp[x, ...])
        E2D_Step2 += evalPTS(gtPTS_RC_rhp[x, ...], PredPTS_Ref2_rhp[x, ...])
        E2D_Step1 += evalPTS(gtPTS_RC_rhp[x, ...], predPTS_Ref1_RC_rhp[x, ...])

        E3D_Step1 += evalPTS(gtPTS_RC_rhp[x, ...], KeyPTS_PJ_1_rhp[x, ...])
        E3D_Step1_RV += evalPTS(gtPTS_RC_rhp[x, ...], KeyPTS_PJ_1_ROT_rhp[x, ...])
        E3D_Step2 += evalPTS(gtPTS_RC_rhp[x, ...], KeyPTS_PJ_2_rhp[x, ...])
        E3D_Step2_RV += evalPTS(gtPTS_RC_rhp[x, ...], KeyPTS_PJ_2_ROT_rhp[x, ...])
        E3D_Step3 += evalPTS(gtPTS_RC_rhp[x, ...], KeyPTS_PJ_3_rhp[x, ...])
        E3D_Step3_RV += evalPTS(gtPTS_RC_rhp[x, ...], KeyPTS_PJ_3_ROT_rhp[x, ...])


        # E3D_Step3 += evalPTS(gtPTS_RC_rhp[x, ...], KeyPTS_PJ_2_rhp[x, ...])

    # print 'current testing erorr 3D:', Error3D/TestLens
    # print 'current testing error 2D:', Error2D/TestLens
    print '____________________________testing  evaluated__________________________________________________'
    print  'step1: 3d: ', E3D_Step1 / TestLens, '3d_rv: ', E3D_Step1_RV / TestLens, '2d: ', E2D_Step1 / TestLens
    print  'step2: 3d: ', E3D_Step2 / TestLens, '3d_rv: ', E3D_Step2_RV / TestLens, '2d: ', E2D_Step2 / TestLens
    print  'step3: 3d: ', E3D_Step3 / TestLens, '3d_rv: ', E3D_Step3_RV / TestLens, '2d: ', E2D_Step3 / TestLens
    print '____________________________testing  end__________________________________________________'
    #



# test300W(SavePath='./R2ID_GRLF_FitEXP/Test',isSave=False,batch_size=1,TestLens=10)
# sgdED_local = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9)
# sgdED_global = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9)
# SLF_LocalNet.compile(loss='mean_squared_error',metrics=['accuracy', final_pred],optimizer=sgdED_local)
# SFL_GlobalNet.compile(loss =  'mean_squared_error',loss_weights=[1,100,0.1],metrics=['accuracy', final_pred],optimizer=sgdED_global)
# # sgdED_local2 = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9)
# # sgdED_global2 = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9)
# # SLF_LocalNet2.compile(loss='mean_squared_error',metrics=['accuracy', final_pred],optimizer=sgdED_local2)
# # SFL_GlobalNet2.compile(loss =  'mean_squared_error',loss_weights=[1,100,0.1],metrics=['accuracy', final_pred],optimizer=sgdED_global2)
# # modelSLF_Local.compile()
# train_epoch(1,SavePath='./R2ID_GRLF_FitEXP',MaxIters=80000)
# test300W(SavePath='./R2ID_GRLF_FitEXP/Test',isSave=False,batch_size=1,TestLens=len(DataTe))

# test300W(SavePath='./R2ID_GRLF_FitEXP/Test',isSave=False,batch_size=1,TestLens=10)
sgdED_local = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9)
sgdED_global = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9)
SLF_LocalNet2.compile(loss='mean_squared_error',metrics=['accuracy', final_pred],optimizer=sgdED_local)
SFL_GlobalNet2.compile(loss =  'mean_squared_error',loss_weights=[1,100,0.1],metrics=['accuracy', final_pred],optimizer=sgdED_global)
# sgdED_local2 = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9)
# sgdED_global2 = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9)
# SLF_LocalNet2.compile(loss='mean_squared_error',metrics=['accuracy', final_pred],optimizer=sgdED_local2)
# SFL_GlobalNet.compile(loss =  'mean_squared_error',loss_weights=[1,100,0.1],metrics=['accuracy', final_pred],optimizer=sgdED_global2)
# modelSLF_Local.compile()
train_epoch(1,SavePath='./R2ID_GRLF_FitEXP',MaxIters=80100)
test300W(SavePath='./R2ID_GRLF_FitEXP/Test',isSave=False,batch_size=1,TestLens=len(DataTe))


sgdED_local = optimizers.SGD(lr=0.00001, decay=1e-6, momentum=0.9)
sgdED_global = optimizers.SGD(lr=0.00001, decay=1e-6, momentum=0.9)
SLF_LocalNet2.compile(loss='mean_squared_error',metrics=['accuracy', final_pred],optimizer=sgdED_local)
SFL_GlobalNet2.compile(loss =  'mean_squared_error',loss_weights=[1,100,0.1],metrics=['accuracy', final_pred],optimizer=sgdED_global)
# modelSLF_Local.compile()
train_epoch(1,SavePath='./R2ID_GRLF_FitEXP',MaxIters=80100)
test300W(SavePath='./R2ID_GRLF_FitEXP/Test',isSave=False,batch_size=1,TestLens=len(DataTe))