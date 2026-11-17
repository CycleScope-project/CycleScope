// cublaslt_case: enumerate/run cublasLt GEMM algo configs, splitK=1.
// --dtype bf16|fp16|fp32|tf32|fp8|int8 --layout NN|NT|TN|TT --m M --n N --k K
// --algo-index i | --list | --default ; --warmups W (default 3)
// --default: first cublasLtMatmulAlgoGetHeuristic result as-is (no splitK=1).
// --algo-shape SM,SN,SK: run the same heuristic as --default on (SM,SN,SK),
//   take its first result, AlgoCheck it on the exec (--m/--n/--k) layouts,
//   then execute it there.  AlgoCheck failure prints the reason and rc=3.
// --sm-count-target N: sets CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET on the
//   op descriptor after the heuristic call, before AlgoCheck/matmul —
//   keeps the heuristic-chosen algo, caps the execution grid plan.
// --sm-count-target-heur N: sets it before the heuristic, so algo
//   selection itself sees the smaller SM count.  Both may be given.
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
static void ck(cudaError_t e,const char*w){if(e){fprintf(stderr,"cuda %s %s\n",w,cudaGetErrorString(e));exit(2);}}
static void ck(cublasStatus_t s,const char*w){if(s){fprintf(stderr,"cublasLt %s %d\n",w,(int)s);exit(2);}}

struct Layouts {
  cublasLtMatrixLayout_t lA,lB,lC,lD;
};

int main(int argc,char**argv){
  std::string dt="bf16",ly="NN"; long m=128,n=128,k=16384; int idx=-1,W=3; bool list=false,deflt=false;
  long sm=-1,sn=-1,sk=-1; int smt=0,smt_heur=0,batch=1,abatch=0; bool splitk1=false; std::string epi="none"; float beta_f=0.0f;
  for(int i=1;i<argc;i++){std::string a=argv[i];auto nx=[&]{return argv[++i];};
    if(a=="--dtype")dt=nx();else if(a=="--layout")ly=nx();else if(a=="--m")m=atol(nx());
    else if(a=="--n")n=atol(nx());else if(a=="--k")k=atol(nx());else if(a=="--algo-index")idx=atoi(nx());
    else if(a=="--warmups")W=atoi(nx());else if(a=="--list")list=true;else if(a=="--default")deflt=true;
    else if(a=="--algo-shape"){std::string s=nx(); if(sscanf(s.c_str(),"%ld,%ld,%ld",&sm,&sn,&sk)!=3){fprintf(stderr,"bad --algo-shape %s\n",s.c_str());return 2;}}
    else if(a=="--sm-count-target")smt=atoi(nx());else if(a=="--sm-count-target-heur")smt_heur=atoi(nx());
    else if(a=="--batch")batch=atoi(nx());else if(a=="--algo-batch")abatch=atoi(nx());else if(a=="--epilogue")epi=nx();else if(a=="--beta")beta_f=atof(nx());
    else if(a=="--splitk1")splitk1=true;
    else {fprintf(stderr,"unknown arg %s\n",a.c_str());return 2;}}
  cublasLtHandle_t lt; ck(cublasLtCreate(&lt),"create");
    cudaDataType_t AT,BT,CT,DT_=(cudaDataType_t)0; cublasComputeType_t CT_; cudaDataType_t ST=CUDA_R_32F;
  if(dt=="bf16"){AT=BT=CUDA_R_16BF;CT=CUDA_R_16BF;CT_=CUBLAS_COMPUTE_32F;}
  else if(dt=="fp16"){AT=BT=CUDA_R_16F;CT=CUDA_R_16F;CT_=CUBLAS_COMPUTE_32F;}
  else if(dt=="fp32"||dt=="tf32"){AT=BT=CUDA_R_32F;CT=CUDA_R_32F;CT_=dt=="tf32"?CUBLAS_COMPUTE_32F_FAST_TF32:CUBLAS_COMPUTE_32F;}
  else if(dt=="fp8"){AT=BT=CUDA_R_8F_E4M3;CT=CUDA_R_16BF;CT_=CUBLAS_COMPUTE_32F;}
  else if(dt=="fp8e5m2"){AT=CUDA_R_8F_E5M2;BT=CUDA_R_8F_E4M3;CT=CUDA_R_16BF;CT_=CUBLAS_COMPUTE_32F;}
  else if(dt=="fp8out"){AT=BT=CUDA_R_8F_E4M3;CT=CUDA_R_8F_E4M3;DT_=CUDA_R_8F_E4M3;CT_=CUBLAS_COMPUTE_32F;}
  else if(dt=="mxfp8"){AT=BT=CUDA_R_8F_E4M3;CT=CUDA_R_16BF;CT_=CUBLAS_COMPUTE_32F;}
  else if(dt=="nvfp4"){AT=BT=CUDA_R_4F_E2M1;CT=CUDA_R_16BF;CT_=CUBLAS_COMPUTE_32F;}
  else if(dt=="fp64"){AT=BT=CT=CUDA_R_64F;CT_=CUBLAS_COMPUTE_64F;ST=CUDA_R_64F;}
  else if(dt=="int8"){AT=BT=CUDA_R_8I;CT=CUDA_R_32I;CT_=CUBLAS_COMPUTE_32I;ST=CUDA_R_32I;}
  else {fprintf(stderr,"bad dtype\n");return 2;}
  // block scaling (mxfp8: VEC32_UE8M0, nvfp4: VEC16_UE4M3)
  cublasLtMatmulMatrixScale_t scale_mode=CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
  void *sa_dev=nullptr,*sb_dev=nullptr;
  if(dt=="mxfp8"||dt=="nvfp4"){
    scale_mode = dt=="mxfp8" ? CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0
                             : CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    long esz = dt=="mxfp8" ? k/32 : k/16;
    ck(cudaMalloc(&sa_dev,(size_t)m*esz),"scaleA");
    ck(cudaMalloc(&sb_dev,(size_t)n*esz),"scaleB");
    ck(cudaMemset(sa_dev,0x7f,(size_t)m*esz),"ssA");
    ck(cudaMemset(sb_dev,0x7f,(size_t)n*esz),"ssB");}
  void *amax_dev=nullptr;
  if(dt=="fp8out") ck(cudaMalloc(&amax_dev,8),"amax");
  void *bias_dev=nullptr,*aux_dev=nullptr;
  cublasLtMatmulDesc_t op; ck(cublasLtMatmulDescCreate(&op,CT_,ST),"desc");

  cublasOperation_t TA= (ly[0]=='N')?CUBLAS_OP_N:CUBLAS_OP_T;
  cublasOperation_t TB= (ly[1]=='N')?CUBLAS_OP_N:CUBLAS_OP_T;
  ck(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&TA,sizeof TA),"ta");
  ck(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSB,&TB,sizeof TB),"tb");
  if(scale_mode!=CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F){
    ck(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_A_SCALE_MODE,&scale_mode,sizeof scale_mode),"asmode");
    ck(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_B_SCALE_MODE,&scale_mode,sizeof scale_mode),"bsmode");
    ck(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,&sa_dev,sizeof sa_dev),"aptr");
    ck(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,&sb_dev,sizeof sb_dev),"bptr");}
  if(dt=="fp8"||dt=="fp8e5m2"||dt=="fp8out"){
    unsigned char fast=1; // FP8 GEMMs require CUBLASLT_MATMUL_DESC_FAST_ACCUM
    ck(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_FAST_ACCUM,&fast,sizeof fast),"fast");
    float one=1.0f;
    ck(cudaMalloc(&sa_dev,4),"ascale"); ck(cudaMemcpy(sa_dev,&one,4,cudaMemcpyHostToDevice),"asc_c");
    if(!sb_dev) ck(cudaMalloc(&sb_dev,4),"bscale"); ck(cudaMemcpy(sb_dev,&one,4,cudaMemcpyHostToDevice),"bsc_c");
    ck(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,&sa_dev,sizeof sa_dev),"ascale");
    ck(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,&sb_dev,sizeof sb_dev),"bscale");}
  if(dt=="fp8out"){
    float dsc=1.0f; void*dscale_dev=nullptr; ck(cudaMalloc(&dscale_dev,4),"dscale"); ck(cudaMemcpy(dscale_dev,&dsc,4,cudaMemcpyHostToDevice),"dsc_c");
    ck(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_D_SCALE_POINTER,&dscale_dev,sizeof dscale_dev),"dscale");
    ck(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_AMAX_D_POINTER,&amax_dev,sizeof amax_dev),"amax");}
  { // epilogue
    cublasLtEpilogue_t e=CUBLASLT_EPILOGUE_DEFAULT;
    if(epi=="bias")e=CUBLASLT_EPILOGUE_BIAS;
    else if(epi=="relu_bias")e=CUBLASLT_EPILOGUE_RELU_BIAS;
    else if(epi=="gelu_bias")e=CUBLASLT_EPILOGUE_GELU_BIAS;
    else if(epi=="gelu_aux_bias")e=CUBLASLT_EPILOGUE_GELU_AUX_BIAS;
    if(e!=CUBLASLT_EPILOGUE_DEFAULT){
      size_t eszb = CT==CUDA_R_64F?8:CT==CUDA_R_32F||CT==CUDA_R_32I?4:2;
      ck(cudaMalloc(&bias_dev,(size_t)n*eszb),"bias");
      ck(cudaMemset(bias_dev,0,(size_t)n*eszb),"sbias");
      ck(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_EPILOGUE,&e,sizeof e),"epi");
      ck(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_BIAS_POINTER,&bias_dev,sizeof bias_dev),"biasptr");
      if(e==CUBLASLT_EPILOGUE_GELU_AUX_BIAS){
        ck(cudaMalloc(&aux_dev,(size_t)m*n*eszb),"aux");
        ck(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER,&aux_dev,sizeof aux_dev),"auxptr");
        if(DT_!=0) ck(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE,&DT_,sizeof DT_),"auxdt");}}}
  auto set_smt=[&](int v,const char*w){
    if(v>0) ck(cublasLtMatmulDescSetAttribute(
        op,CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET,&v,sizeof v),w);};
  // AUX_LD = leading dim of the aux output = rows of D of the active shape
  auto set_auxld=[&](long rows){
    if(epi=="gelu_aux_bias")
      ck(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD,&rows,sizeof rows),"auxld");};
  set_auxld(m);
  // column-major ld under op semantics: A op is (m x k): ld = rows of stored matrix
  auto make_layouts=[&](long mm,long nn,long kk,Layouts&L,bool al_arg=false){ L.lD=nullptr;
    unsigned long lda = TA==CUBLAS_OP_N? mm:kk, ldb = TB==CUBLAS_OP_N? kk:nn, ldc=mm;
    ck(cublasLtMatrixLayoutCreate(&L.lA,AT,lda,TA==CUBLAS_OP_N?kk:mm,lda),"lA");
    ck(cublasLtMatrixLayoutCreate(&L.lB,BT,ldb,TB==CUBLAS_OP_N?nn:kk,ldb),"lB");
    ck(cublasLtMatrixLayoutCreate(&L.lC,CT,mm,nn,ldc),"lC");
    if(DT_!=0) ck(cublasLtMatrixLayoutCreate(&L.lD,DT_,mm,nn,ldc),"lD");
    else L.lD=L.lC;
    int lb=(al_arg&&abatch>0?abatch:batch);
    if(lb>1){
      for(auto*lp:{&L.lA,&L.lB,&L.lC}){
        ck(cublasLtMatrixLayoutSetAttribute(*lp,CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,&lb,sizeof lb),"bcount");}
      long long sA=(long long)lda*(TA==CUBLAS_OP_N?kk:mm), sB=(long long)ldb*(TB==CUBLAS_OP_N?nn:kk), sC=(long long)mm*nn;
      ck(cublasLtMatrixLayoutSetAttribute(L.lA,CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,&sA,sizeof sA),"sA");
      ck(cublasLtMatrixLayoutSetAttribute(L.lB,CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,&sB,sizeof sB),"sB");
      ck(cublasLtMatrixLayoutSetAttribute(L.lC,CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,&sC,sizeof sC),"sC");}
  };
  Layouts ex,al;
  make_layouts(m,n,k,ex);
  size_t eszA = AT==CUDA_R_32F?4:AT==CUDA_R_8I||AT==CUDA_R_8F_E4M3?1:2;
  size_t eszC = CT==CUDA_R_32F||CT==CUDA_R_32I?4:2;
  void *da,*db,*dc,*dd=nullptr;
  size_t eszD = (DT_==0)?eszC:1; ck(cudaMalloc(&da,(size_t)m*k*eszA*batch),"a"); ck(cudaMalloc(&db,(size_t)k*n*eszA*batch),"b"); ck(cudaMalloc(&dc,(size_t)m*n*eszC*batch),"c"); if(DT_!=0){ck(cudaMalloc(&dd,(size_t)m*n*eszD*batch),"d");ck(cudaMemset(dd,0,(size_t)m*n*eszD*batch),"sd");} else dd=dc;
  ck(cudaMemset(da,1,(size_t)m*k*eszA),"sa"); ck(cudaMemset(db,1,(size_t)k*n*eszA),"sb"); ck(cudaMemset(dc,0,(size_t)m*n*eszC*batch),"sc");
  float alpha=1.0f,beta=beta_f; int ialpha=1,ibeta=(int)beta_f; double dalpha=1.0,dbeta=(double)beta_f;
  void *workspace; size_t ws=256u<<20; ck(cudaMalloc(&workspace,ws),"ws");
  cudaStream_t st=0;
  cublasLtMatmulHeuristicResult_t hr[64]; int cnt=0;
  cublasLtMatmulPreference_t pref; ck(cublasLtMatmulPreferenceCreate(&pref),"pref");
  ck(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&ws,sizeof ws),"prefws");
  std::vector<cublasLtMatmulHeuristicResult_t> ok;
  set_smt(smt_heur,"smt-heur");
  if(sm>0){
    // heuristic on the algo shape, first result as-is (same rule as --default)
    make_layouts(sm,sn,sk,al,true);
    set_auxld(sm);
    ck(cublasLtMatmulAlgoGetHeuristic(lt,op,al.lA,al.lB,al.lC,al.lC,pref,64,hr,&cnt),"heur-shape");
    if(cnt<1){fprintf(stderr,"algo-shape heuristic returned no algo\n");return 3;}
    cublasLtMatmulHeuristicResult_t h=hr[0];
    if(splitk1){ // force single-k reduction on the picked algo
      int split=1; cublasLtMatmulAlgoConfigSetAttribute(&h.algo,CUBLASLT_ALGO_CONFIG_SPLITK_NUM,&split,sizeof split);
      int nores=1; cublasLtMatmulAlgoConfigSetAttribute(&h.algo,CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,&nores,sizeof nores);
    }
    set_smt(smt,"smt");
    set_auxld(m);
    cublasLtMatmulHeuristicResult_t chk{};
    cublasStatus_t cs=cublasLtMatmulAlgoCheck(lt,op,ex.lA,ex.lB,ex.lC,ex.lC,&h.algo,&chk);
    if(cs!=CUBLAS_STATUS_SUCCESS){
      fprintf(stderr,"algo_check failed on exec shape %ld,%ld,%ld (algo-shape %ld,%ld,%ld): status %d\n",m,n,k,sm,sn,sk,(int)cs);
      return 3;
    }
    ok.push_back(chk);
    idx=0;
  } else if(deflt){
    cublasStatus_t hs=cublasLtMatmulAlgoGetHeuristic(lt,op,ex.lA,ex.lB,ex.lC,ex.lC,pref,64,hr,&cnt);
    if(hs!=CUBLAS_STATUS_SUCCESS){ // retry without workspace preference (fp64 big shapes can hit heur 7)
      fprintf(stderr,"heur status %d, retry w/o pref\n",(int)hs);
      cublasLtMatmulPreference_t pref2; ck(cublasLtMatmulPreferenceCreate(&pref2),"pref2");
      ck(cublasLtMatmulAlgoGetHeuristic(lt,op,ex.lA,ex.lB,ex.lC,ex.lC,pref2,64,hr,&cnt),"heur2");}
    if(cnt<1){fprintf(stderr,"heuristic returned no algo\n");return 2;}
    set_smt(smt,"smt");
    ok.push_back(hr[0]);
    idx=0;
  } else {
    ck(cublasLtMatmulAlgoGetHeuristic(lt,op,ex.lA,ex.lB,ex.lC,ex.lC,pref,64,hr,&cnt),"heur");
    set_smt(smt,"smt");
    for(int i=0;i<cnt;i++){
      cublasLtMatmulHeuristicResult_t h=hr[i]; int split=1;
      cublasLtMatmulAlgoConfigSetAttribute(&h.algo,CUBLASLT_ALGO_CONFIG_SPLITK_NUM,&split,sizeof split);
      int nores=1; // CUBLASLT_REDUCTION_SCHEME_NONE
      cublasLtMatmulAlgoConfigSetAttribute(&h.algo,CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,&nores,sizeof nores);
      if(cublasLtMatmulAlgoCheck(lt,op,ex.lA,ex.lB,ex.lC,ex.lC,&h.algo,&h)==CUBLAS_STATUS_SUCCESS) ok.push_back(h);
    }
  }
  if(list){for(auto&h:ok){int t=0,s=0;unsigned int cl=0;
      cublasLtMatmulAlgoConfigGetAttribute(&h.algo,CUBLASLT_ALGO_CONFIG_TILE_ID,&t,sizeof t,nullptr);
      cublasLtMatmulAlgoConfigGetAttribute(&h.algo,CUBLASLT_ALGO_CONFIG_STAGES_ID,&s,sizeof s,nullptr);
      cublasLtMatmulAlgoConfigGetAttribute(&h.algo,CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID,&cl,sizeof cl,nullptr);
      printf("algo tile=%d stages=%d cluster=%u\n",t,s,cl);} printf("accepted=%zu\n",ok.size()); return 0;}
  if(idx<0||idx>=(int)ok.size()){fprintf(stderr,"algo-index out of range %d/%zu\n",idx,ok.size());return 2;}
  // fp64: heuristic algos may pass AlgoCheck yet fail at exec (status 14);
  // fall through the accepted list until one launches.
  {int id=-1,t=0,s=0,sp=0,sw=0;unsigned int cl=0;
   cublasLtMatmulAlgoConfigGetAttribute(&ok[idx].algo,CUBLASLT_ALGO_CONFIG_ID,&id,sizeof id,nullptr);
   cublasLtMatmulAlgoConfigGetAttribute(&ok[idx].algo,CUBLASLT_ALGO_CONFIG_TILE_ID,&t,sizeof t,nullptr);
   cublasLtMatmulAlgoConfigGetAttribute(&ok[idx].algo,CUBLASLT_ALGO_CONFIG_STAGES_ID,&s,sizeof s,nullptr);
   cublasLtMatmulAlgoConfigGetAttribute(&ok[idx].algo,CUBLASLT_ALGO_CONFIG_SPLITK_NUM,&sp,sizeof sp,nullptr);
   cublasLtMatmulAlgoConfigGetAttribute(&ok[idx].algo,CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,&sw,sizeof sw,nullptr);
   cublasLtMatmulAlgoConfigGetAttribute(&ok[idx].algo,CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID,&cl,sizeof cl,nullptr);
   printf("algo id=%d tile=%d stages=%d splitK=%d swizzle=%d cluster=%u\n",id,t,s,sp,sw,cl);}
  void*al_=(CT_==CUBLAS_COMPUTE_32I)?(void*)&ialpha:(CT_==CUBLAS_COMPUTE_64F)?(void*)&dalpha:(void*)&alpha;
  void*be=(CT_==CUBLAS_COMPUTE_32I)?(void*)&ibeta:(CT_==CUBLAS_COMPUTE_64F)?(void*)&dbeta:(void*)&beta;
  auto mm=[&]{cublasStatus_t s=cublasLtMatmul(lt,op,al_,da,ex.lA,db,ex.lB,be,dc,ex.lC,dd,ex.lD,&ok[idx].algo,workspace,ws,st);ck(s,"matmul");};
  // exec-status fallback: if an accepted algo fails at launch, walk the
  // rest of the heuristic list INSIDE the warmup call (launch count stays W+1)
  auto mm2=[&]{cublasStatus_t s0=cublasLtMatmul(lt,op,al_,da,ex.lA,db,ex.lB,be,dc,ex.lC,dd,ex.lD,&ok[idx].algo,workspace,ws,st);
    while(s0!=CUBLAS_STATUS_SUCCESS && idx+1<(int)ok.size()){
      fprintf(stderr,"algo %d exec status %d, trying %d\n",idx,(int)s0,idx+1); idx++;
      s0=cublasLtMatmul(lt,op,al_,da,ex.lA,db,ex.lB,be,dc,ex.lC,dd,ex.lD,&ok[idx].algo,workspace,ws,st);}
    if(s0!=CUBLAS_STATUS_SUCCESS){fprintf(stderr,"matmul exec status %d\n",(int)s0);exit(3);}};
  for(int i=0;i<W;i++)mm2();
  mm(); ck(cudaDeviceSynchronize(),"sync");
  printf("done algo=%d\n",idx); return 0;
}
