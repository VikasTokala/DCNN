/* [e,filters] = compute_delay1_mex(y,speech,L,mu,option,iter,sub,algo)
/* 
/* Computes time-delays between 2 different microphone signals 
/* Technique: adaptive eigenvalue decomposition (J. Benesty), no subsampling possible
/* 
/* AUTHOR	Simon Doclo, 11/01/01
/* OUTPUT	delays        : delays between microphone signals and 1st microphone signal (in samples)
/*               filters       : different estimated impulse responses
/*               e             : error signal
/*               corr          : correlation between filters
/* INPUTS        y             : microphone signals
/*               speech        : speech/noise detection per sample (only adaptation during speech)
/*               L             : lengths of the different filters
/*               mu            : stepsize parameter for LMS-procedure (can be larger than 2!) 
/*               option        : normalisation (0: no normalisation - default, 1: normalisation per channel, 2: global normalisation)
/*               iter          : number of iterations (optional, default 1)
/*               sub           : subsampling factor (optional, default 1)
/*               algo          : 1 = LMS-based subspace tracking procedure (smallest eigenvalue is zero) (default)
/*                               2 = LMS-based subspace tracking procedure */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <limits.h>
#include <float.h>
#include "mex.h"
#include "matrix.h"

#define EP 0.0000000001

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  double *in,*speech,*e,*filter,*out;
  double *tmp_in1,*tmp_in2,*tmp_filt1,*tmp_filt2;
  int signallength,L,option,iter;
  double mu,filt,filtmu,norm,tmp,en1,en2,en;
  int i,j,k,start,sub,algo;

  /* Input collection */ 
  signallength = (int) mxGetM(prhs[0]);
  in = (double *) mxGetPr(prhs[0]);
  speech = (double *) mxGetPr(prhs[1]);
  L = (int) *mxGetPr(prhs[2]);
  mu = (double) *mxGetPr(prhs[3]);
  if (nrhs > 4) {option = (int) *mxGetPr(prhs[4]);}
  else {option = 0;}
  if (nrhs > 5) {iter = (int) *mxGetPr(prhs[5]);}
  else {iter = 1;}
  if (nrhs > 6) {sub = (int) *mxGetPr(prhs[6]);}
  else {sub = 1;}
  if (nrhs > 7) {algo = (int) *mxGetPr(prhs[7]);}
  else {algo = 1;}

  /* Output allocation */
  plhs[0] = (mxArray *) mxCreateDoubleMatrix(floor(((double) signallength)/sub),1,mxREAL);
  e = (double *) mxGetPr(plhs[0]); 
  plhs[1] = (mxArray *) mxCreateDoubleMatrix(2*L,floor(((double) signallength)/sub),mxREAL);
  out = (double *) mxGetPr(plhs[1]); 
  filter = (double *) mxCalloc(2*L,sizeof(double));
  
  /*Initialisation */
  tmp_filt1 = filter;
  for (i=0; i<2*L; i++) {*tmp_filt1++ = 0;}
  filter[L/2-1] = 1;

  start = ceil(((double) L)/sub)*sub;

  in += start-1;
  speech += start-1;
  e += start/sub-1;

  /* Algorithm */
  for (i=start;i<=signallength;i+=sub)
  {
      if (*speech == 1) {
	
	/* Normalisation */
	if (option == 0) {
	  en1 = 1;
	  en2 = 1;
	}
	else {
	  en1 = 0;
	  en2 = 0;
	  tmp_in1 = in+signallength;  
	  tmp_in2 = in;  
	  for (j=0;j<L;j++) {
	    tmp = *tmp_in1--;
	    en1 += tmp*tmp;
	    tmp = *tmp_in2--;
	    en2 += tmp*tmp;
	  }	  
	  if (option == 1) {
	    en1 = 1/(sqrt(en1)+EP);
	    en2 = 1/(sqrt(en2)+EP); 
	  }
	  if (option == 2) {
	    tmp = 1/(sqrt(en1+en2)+EP);
	    en1=tmp;
	    en2=tmp;
	  }
	}

	/* Iteration */
	for (k=0; k < iter; k++) {

	  filt = 0;  
	  tmp_filt1 = filter;
	  tmp_filt2 = filter+L;
	  tmp_in1 = in+signallength; /* x2 */ 
	  tmp_in2 = in; /* x1 */  
	  for (j=0;j<L;j++) {	  
	    filt += (*tmp_in2--)*(*tmp_filt2++)*en2 - (*tmp_in1--)*(*tmp_filt1++)*en1;	  
	  }	
	  *e = filt;
	  filtmu = filt*mu;

	  tmp_filt1 = filter;
	  tmp_filt2 = filter+L;
	  tmp_in1 = in+signallength;  /* x2 */ 
	  tmp_in2 = in; /* x1 */ 
	  if (algo == 1) {	    
	    for (j=0;j<L;j++) { 
	      *tmp_filt1 += filtmu*(*tmp_in1--)*en1;
	      tmp_filt1++;
	      *tmp_filt2 -= filtmu*(*tmp_in2--)*en2;
	      tmp_filt2++;
	    }
	  }
	  if (algo == 2) {
	    for (j=0;j<L;j++) { 
	      *tmp_filt1 += filtmu*((*tmp_in1--)*en1+(*tmp_filt1)*filt);
	      tmp_filt1++;
	      *tmp_filt2 -= filtmu*((*tmp_in2--)*en2-(*tmp_filt2)*filt);
	      tmp_filt2++;
	    }
	  }

	  norm = 0;
	  tmp_filt1 = filter;
	  for (j=0;j<2*L;j++) {
	    tmp = *tmp_filt1++;
	    norm += tmp*tmp;
	  }
	  norm = 1/sqrt(norm);

	  tmp_filt1 = filter;
	  for (j=0;j<2*L;j++) { 
	    *tmp_filt1 *= norm;
	    tmp_filt1++;
	  }
	}

	/* Copy filter to out */
	tmp_filt1 = filter;
	tmp_filt2 = out+(i/sub-1)*2*L;
	for (j=0;j<2*L;j++) {
	  *tmp_filt2++ = *tmp_filt1++;
	}
      }

      speech += sub;
      in += sub;
      e++; 
  }

  mxFree(filter);
}
