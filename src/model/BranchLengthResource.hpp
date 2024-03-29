#ifndef _BRANCH_LENGTH_RESOURCE
#define _BRANCH_LENGTH_RESOURCE

#include "pll.h"

#include <vector>

class TreeAln; 

class BranchLengthResource
{
  friend class TreeAln;  	// this is a resouce class only; friendship is okay 

public: 
  void initialize(nat numTax, nat numPart ); 
  void assign(TreeAln &traln) ;  

private: 
  nat _numTax; 
  nat _numPart; 
  
  std::vector<double> _zqr  ;
  std::vector<double> _currentZQR  ;
  std::vector<double> _currentLZR  ;
  std::vector<double> _currentLZQ  ;
  std::vector<double> _currentLZS  ;
  std::vector<double> _currentLZI  ;
  std::vector<double> _lzs  ;
  std::vector<double> _lzq  ;
  std::vector<double> _lzr  ;
  std::vector<double> _lzi  ;
  std::vector< std::vector<double> > _qz; 
  std::vector< std::vector<double> > _rz; 
  std::vector< std::vector<double> > _z; 
  
  std::vector<double> _parameterValues; 
}; 

#endif
