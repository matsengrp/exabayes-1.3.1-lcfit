#include "ParameterContent.hpp"
#include "model/Branch.hpp"
#include <limits>
#include <algorithm> 



ParameterContent::ParameterContent(std::vector<double> valuesI, std::vector<BranchPlain> topoI,
				   std::vector<BranchLength> blI, std::vector<ProtModel>  pmI) 
  : values{valuesI}
  , topology{topoI}
  , branchLengths{blI}
  , protModel{pmI}
  {
  }  



void ParameterContent::deserialize( std::istream &in )  
{
  for(auto &v : values)
    v = cRead<double>(in); 

  for(auto &b : branchLengths)
    b.deserialize(in);

  for(auto &b : topology)
    b.deserialize(in); 
  
  for(auto &v :  protModel)
    v = ProtModel(cRead<int>(in));
} 


void ParameterContent::serialize( std::ostream &out) const 
{
  for(auto &v : values)
    cWrite(out, v); 

  for(auto &b : branchLengths)    
    b.serialize(out);

  for(auto &b : topology)
    b.serialize(out); 

  for(auto &v : protModel)
    {
      auto tmp = int(v);
      cWrite<int>(out,tmp); 
    }
}   



std::ostream& operator<<(std::ostream& out, const ParameterContent &rhs)
{
  auto isFirst = bool{true}; 
 

  if(rhs.values.size() > 0)
    {
      for(auto &v : rhs.values)
	{
	  out << (isFirst ? "" : ","  )<< v; 
	  isFirst = false; 
	}
    }
  else if(rhs.branchLengths.size( )> 0)
    {
      for(auto &b : rhs.branchLengths)
	{
	  out << (isFirst ? "" : ",") << b ; 
	  isFirst = false; 
	}
    }
  else if(rhs.topology.size() > 0)
    {
      for(auto &b : rhs.topology)
	{
	  out << (isFirst ? "" : ",") << b; 
	  isFirst = false; 
	}
    }
  else if(rhs.protModel.size()  > 0)
    {
      for(auto &p : rhs.protModel)
      	out << p ; 
    }
  else     
    assert(0); 

  return out; 
} 
