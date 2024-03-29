#ifndef _PARAMETER_CONTENT
#define _PARAMETER_CONTENT

#include <iostream>
#include <vector>

#include "model/ProtModel.hpp"
#include "model/Branch.hpp"
#include "system/Serializable.hpp"



// for really properly implementing this, we'd finally need a type for
// internal and absolute branch lengths (different types of branches). 
// Until then: I know, it's suboptimal 

class  ParameterContent : public Serializable
{
public: 
  ParameterContent(std::vector<double> valuesI = std::vector<double>{}, 
		   std::vector<BranchPlain> topoI = std::vector<BranchPlain>{}, 
		   std::vector<BranchLength> blI = std::vector<BranchLength>{}, 
		   std::vector<ProtModel>  pmI = std::vector<ProtModel>{}) ; 

public:   			// public stuff that should be private 
  std::vector<double> values; 
  std::vector<BranchPlain> topology;
  std::vector<BranchLength> branchLengths; 
  std::vector<ProtModel> protModel; 
  
  virtual void deserialize( std::istream &in )  ; 
  virtual void serialize( std::ostream &out)  const;   

  friend std::ostream& operator<<(std::ostream& out,  const ParameterContent &rhs); 
}; 

#endif
