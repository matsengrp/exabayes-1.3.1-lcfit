#ifndef _CATEGORIES_H
#define _CATEGORIES_H

#include <vector>
#include <string>
#include <memory>
#include "parameters/AbstractParameter.hpp"

enum class Category :  int  
{  
  TOPOLOGY = 0, 
    BRANCH_LENGTHS = 1, 
    FREQUENCIES = 2,
    SUBSTITUTION_RATES = 3,
    RATE_HETEROGENEITY = 4,	
    AA_MODEL= 5  

} ; 


namespace std
{
  template<> struct less<Category>
  {
    bool operator()(const Category& a, const Category& b) const 
    {
      return  int(a) < int(b); 
    }
  };
    
  template<> struct hash<Category>
  {
    size_t operator()(const Category& a) const 
    {
      return std::hash<size_t>()(size_t(a)); 
    }
  }; 
}

namespace CategoryFuns 
{
  /** 
      @brief gets a list of all categories 
   */ 
  std::vector<Category> getAllCategories(); 
  /**
     @brief gets the verbose name of a category 
   */ 
  std::string getLongName(Category cat); 
  /** 
      @brief gets the short name of a category 
   */ 
  std::string getShortName(Category cat); 
  /** 
      @brief gets the name of the prior that is associated with a category 
   */ 
  std::string getPriorName(Category cat); 
  /** 
      @brief gets the category by name  
   */ 
  Category getCategoryByPriorName(std::string name); 
  /** 
      @brief gets the category by name of the linking parameter in the config file 
   */   
  Category getCategoryFromLinkLabel(std::string name); 
  std::unique_ptr<AbstractParameter> getParameterFromCategory(Category cat, nat id, nat idOfMyKind, std::vector<nat> partitions); 
  
} 

std::ostream&  operator<<(std::ostream& out, const Category &rhs); 

#endif
 


