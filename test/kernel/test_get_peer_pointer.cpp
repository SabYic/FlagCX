/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Demo: verify flagcxGetPeerPointer(mem, offset, team, peer) correctness.
 *
 * Uses flagcxIntraTestPeerPointer host wrapper (declared in flagcx_kernel.h,
 * implemented in device_api.cu) which internally launches a kernel that:
 *   - Each rank writes myRank into every peer's buffer at offset [myRank]
 *   - After sync, each rank's buffer should contain buf[i] == i for all i
 *
 * Usage: mpirun -np <nGPUs> ./test_get_peer_pointer [-R 0|1|2]
 *   -R 0: raw (cudaMalloc, no registration)
 *   -R 1: IPC (flagcxMemAlloc + CommRegister)
 *   -R 2: window (flagcxMemAlloc + CommWindowRegister)
 ************************************************************************/

#include "flagcx.h"
#include "flagcx_kernel.h"
#include "tools.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

int main(int argc, char *argv[]) {
  parser args(argc, argv);
  int localRegister = args.getLocalRegister();

  flagcxHandlerGroup_t handler;
  FLAGCXCHECK(flagcxHandleInit(&handler));
  flagcxUniqueId_t &uniqueId = handler->uniqueId;
  flagcxComm_t &comm = handler->comm;
  flagcxDeviceHandle_t &devHandle = handler->devHandle;

  int color = 0;
  int worldSize = 1, worldRank = 0;
  int totalProcs = 1, proc = 0;
  MPI_Comm splitComm;
  uint64_t splitMask = args.getSplitMask();
  initMpiEnv(argc, argv, worldRank, worldSize, proc, totalProcs, color,
             splitComm, splitMask);

  int nGpu;
  FLAGCXCHECK(devHandle->getDeviceCount(&nGpu));
  FLAGCXCHECK(devHandle->setDevice(worldRank % nGpu));

  // Enable P2P access to all other GPUs used by peers in this test.
  // Without this, IPC writes via mapped pointers may not reach the target
  // GPU's DRAM directly (PCIe topology requires explicit P2P enablement).
  for (int i = 0; i < nGpu; i++) {
    if (i == worldRank % nGpu) continue;
    int canAccess = 0;
    cudaDeviceCanAccessPeer(&canAccess, worldRank % nGpu, i);
    if (canAccess) {
      cudaError_t err = cudaDeviceEnablePeerAccess(i, 0);
      if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
        printf("rank%d: WARNING: cudaDeviceEnablePeerAccess(%d) failed: %s\n",
               worldRank, i, cudaGetErrorString(err));
      }
    }
  }

  if (proc == 0)
    FLAGCXCHECK(flagcxGetUniqueId(&uniqueId));
  MPI_Bcast((void *)uniqueId, sizeof(flagcxUniqueId), MPI_BYTE, 0, splitComm);
  MPI_Barrier(MPI_COMM_WORLD);

  FLAGCXCHECK(flagcxCommInitRank(&comm, totalProcs, uniqueId, proc));

  // Buffer: nRanks ints
  size_t bufSize = (size_t)totalProcs * sizeof(int);

  void *regBuff = nullptr;
  void *regHandle = nullptr;
  flagcxWindow_t win = nullptr;
  flagcxDevMem_t devMem = nullptr;

  if (localRegister == 2) {
    FLAGCXCHECK(flagcxMemAlloc(&regBuff, bufSize));
    FLAGCXCHECK(flagcxCommWindowRegister(comm, regBuff, bufSize, &win,
                                         FLAGCX_WIN_COLL_SYMMETRIC));
  } else if (localRegister == 1) {
    FLAGCXCHECK(flagcxMemAlloc(&regBuff, bufSize));
    FLAGCXCHECK(flagcxCommRegister(comm, regBuff, bufSize, &regHandle));
  } else {
    FLAGCXCHECK(
        devHandle->deviceMalloc(&regBuff, bufSize, flagcxMemDevice, NULL));
  }

  // Create device communicator
  flagcxDevCommRequirements reqs = FLAGCX_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.intraBarrierCount = 1;
  flagcxDevComm_t devComm = nullptr;
  FLAGCXCHECK(flagcxDevCommCreate(comm, &reqs, &devComm));

  // Create device memory handle
  FLAGCXCHECK(flagcxDevMemCreate(comm, regBuff, bufSize, win, &devMem));

  flagcxStream_t stream;
  FLAGCXCHECK(devHandle->streamCreate(&stream));

  int *hostBuf = (int *)malloc(bufSize);
  memset(hostBuf, 0, bufSize);

  if (proc == 0 && color == 0) {
    printf("\n# flagcxGetPeerPointer Correctness Test\n");
    printf("# nRanks=%d, regMode=%s\n\n", totalProcs,
           localRegister == 2   ? "window"
           : localRegister == 1 ? "ipc"
                                : "raw");
  }

  // Clear buffer
  FLAGCXCHECK(devHandle->deviceMemset(regBuff, 2, bufSize, flagcxMemDevice, NULL));
  FLAGCXCHECK(devHandle->streamSynchronize(stream));
  MPI_Barrier(MPI_COMM_WORLD);

  // Launch the peer pointer test kernel via host wrapper.
  // The kernel uses bar.sync(AcqRel) internally to ensure all cross-GPU IPC
  // writes are visible before the kernel returns.
  FLAGCXCHECK(flagcxIntraTestPeerPointer(devMem, devComm, stream));
  FLAGCXCHECK(devHandle->streamSynchronize(stream));
  MPI_Barrier(MPI_COMM_WORLD);

  // Flush IPC writes: on PCIe-connected GPUs, remote IPC writes bypass the
  // target GPU's L2 cache. A D2D round-trip through a temp buffer forces the
  // copy engine to read from DRAM, refreshing stale L2 cachelines.
  void *tempBuf = nullptr;
  FLAGCXCHECK(devHandle->deviceMalloc(&tempBuf, bufSize, flagcxMemDevice, NULL));
  FLAGCXCHECK(devHandle->deviceMemcpy(tempBuf, regBuff, bufSize,
                                      flagcxMemcpyDeviceToDevice, NULL));
  FLAGCXCHECK(devHandle->deviceMemcpy(regBuff, tempBuf, bufSize,
                                      flagcxMemcpyDeviceToDevice, NULL));
  FLAGCXCHECK(devHandle->deviceFree(tempBuf, flagcxMemDevice, NULL));

  // Copy back and verify
  FLAGCXCHECK(devHandle->deviceMemcpy(hostBuf, regBuff, bufSize,
                                      flagcxMemcpyDeviceToHost, NULL));

  FLAGCXCHECK(devHandle->streamSynchronize(stream));
  MPI_Barrier(MPI_COMM_WORLD);
  bool pass = true;
  for (int i = 0; i < totalProcs; i++) {
    if (hostBuf[i] != i) {
      printf("  rank%d: FAIL [%d]: got %d expected %d\n", proc, i, hostBuf[i], i);
      pass = false;
    } else {
      printf("  rank%d: OK   [%d]: got %d\n", proc, i, hostBuf[i]);
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  int ok = pass ? 1 : 0;
  MPI_Allreduce(MPI_IN_PLACE, &ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  if (proc == 0 && color == 0) {
    printf("  PeerPointer write test:  %s\n", ok ? "PASS" : "FAIL");
  }

  // =========================================================================
  // Test 2: flagcxIntraVerifyIntraPointer — prints peerPtrs table + self-check
  // =========================================================================
  if (proc == 0 && color == 0) {
    printf("\n# Test 2: VerifyIntraPointer (debug print + getIntraPointer write)\n");
  }

  // Clear buffer
  FLAGCXCHECK(devHandle->deviceMemset(regBuff, 2, bufSize, flagcxMemDevice, NULL));
  FLAGCXCHECK(devHandle->streamSynchronize(stream));
  MPI_Barrier(MPI_COMM_WORLD);

  // Launch verify kernel
  FLAGCXCHECK(flagcxIntraVerifyIntraPointer(devMem, devComm, stream));
  FLAGCXCHECK(devHandle->streamSynchronize(stream));
  MPI_Barrier(MPI_COMM_WORLD);

  // Flush IPC writes
  {
    void *tmp = nullptr;
    FLAGCXCHECK(devHandle->deviceMalloc(&tmp, bufSize, flagcxMemDevice, NULL));
    FLAGCXCHECK(devHandle->deviceMemcpy(tmp, regBuff, bufSize,
                                        flagcxMemcpyDeviceToDevice, NULL));
    FLAGCXCHECK(devHandle->deviceMemcpy(regBuff, tmp, bufSize,
                                        flagcxMemcpyDeviceToDevice, NULL));
    FLAGCXCHECK(devHandle->deviceFree(tmp, flagcxMemDevice, NULL));
  }

  // Copy back and verify
  FLAGCXCHECK(devHandle->deviceMemcpy(hostBuf, regBuff, bufSize,
                                      flagcxMemcpyDeviceToHost, NULL));
  FLAGCXCHECK(devHandle->streamSynchronize(stream));
  MPI_Barrier(MPI_COMM_WORLD);

  bool pass2 = true;
  for (int i = 0; i < totalProcs; i++) {
    if (hostBuf[i] != i) {
      printf("  rank%d: T2 FAIL [%d]: got %d expected %d\n", proc, i, hostBuf[i], i);
      pass2 = false;
    } else {
      printf("  rank%d: T2 OK   [%d]: got %d\n", proc, i, hostBuf[i]);
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  int ok2 = pass2 ? 1 : 0;
  MPI_Allreduce(MPI_IN_PLACE, &ok2, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  if (proc == 0 && color == 0) {
    printf("  VerifyIntraPointer test: %s\n", ok2 ? "PASS" : "FAIL");
  }

  // Final result
  if (proc == 0 && color == 0) {
    printf("\n# Result: %s\n\n", (ok && ok2) ? "ALL PASSED" : "FAILED");
  }

  // Cleanup
  FLAGCXCHECK(devHandle->streamDestroy(stream));
  FLAGCXCHECK(flagcxDevMemDestroy(comm, devMem));
  FLAGCXCHECK(flagcxDevCommDestroy(comm, devComm));

  if (localRegister == 2) {
    FLAGCXCHECK(flagcxCommWindowDeregister(comm, win));
  } else if (localRegister == 1) {
    FLAGCXCHECK(flagcxCommDeregister(comm, regHandle));
  }

  FLAGCXCHECK(flagcxCommDestroy(comm));

  if (localRegister >= 1) {
    FLAGCXCHECK(flagcxMemFree(regBuff));
  } else {
    FLAGCXCHECK(devHandle->deviceFree(regBuff, flagcxMemDevice, NULL));
  }
  free(hostBuf);
  FLAGCXCHECK(flagcxHandleFree(handler));

  MPI_Finalize();
  return 0;
}