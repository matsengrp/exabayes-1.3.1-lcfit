/** 
    @file Path.hpp represents a  path in a tree
    
    Currently we are assuming, the path always is connected. 
 */ 


#ifndef _PATH_H
#define _PATH_H

#include <vector>

#include "model/TreeAln.hpp"
#include "math/Randomness.hpp"
#include "priors/PriorBelief.hpp"
#include "model/Branch.hpp"


class Path
{
public:   
/** @brief returns true, if the node with a given id is part of this branch */ 
  bool nodeIsOnPath(int node) const;  
  /** @brief for all branches in the path, copy over the branch lengths */ 
  void saveBranchLengthsPath(const TreeAln& traln, const std::vector<AbstractParameter*> &params); 

  /** @brief asserts that this path exists in a given tree */ 
  void debug_assertPathExists(TreeAln& traln); 

  /** @brief assigns stored branch lengths of a path to a given tree  */ 
  void restoreBranchLengthsPath(TreeAln &traln, const std::vector<AbstractParameter*> &blParams) const ; 
  /** @brief only add a branch to the path, if it is novel. If the new
      branch cancels out an existing branch, the path is shortened again */ 
  void pushToStackIfNovel(BranchPlain b, const TreeAln &traln ); 

  // straight-forward container methods 
  void append(BranchPlain value); 
  void clear(); 
  /** @brief number of branches in the path */ 
  nat size() const {return stack.size(); }

  /** @brief yields the branch */  
  BranchPlain& at(int num){return stack[num]; }
  BranchPlain at(int num) const{return stack[num];}

  /** @brief reverse the path */ 
  void reverse(); 

  /** @brief removes the last element */ 
  void pop(); 
  /** @brief removes the first element */ 
  void popFront(); 
  

  /** @brief returns the id of the nth node in the path. nodes 0 and n+1 are the outer nodes in this path that do not have a neighbor. */ 
  int getNthNodeInPath(nat num) const ; 
  
  /** @brief gets the number of nodes represented by the path (assuming it is connected)  */
  int getNumberOfNodes() const {return stack.size()  + 1 ;   }
  void printWithBLs(TreeAln &traln ) const; 

  void multiplyBranch(TreeAln &traln, Randomness &rand, BranchLength bl, double parameter, double &hastings, PriorBelief &prior,  AbstractParameter* const param) const ; 

  void findPath(const TreeAln& traln, nodeptr p, nodeptr q);
  friend std::ostream& operator<<(std::ostream &out, const Path &rhs)  ;

private: 
  std::vector<BranchPlain> stack; 
  std::vector<BranchLengths> bls; 

  bool findPathHelper(const TreeAln &traln, nodeptr p, const BranchPlain &target);
}; 

#endif
