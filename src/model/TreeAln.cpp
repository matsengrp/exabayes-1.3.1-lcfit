#include <sstream>
#include <cassert>
#include <cstring>
#include <unordered_set>

#include "system/ByteFile.hpp"

#include "RateHelper.hpp"
#include "parameters/BranchLengthsParameter.hpp"
#include "parameters/AbstractParameter.hpp"
#include "TreeRandomizer.hpp"
#include "TreeAln.hpp"
#include "system/GlobalVariables.hpp"
#include "system/BoundsChecker.hpp"
#include "TreePrinter.hpp"
#include "Partition.hpp"


using std::cout;
using std::endl; 


TreeAln::TreeAln(nat numTax, bool isSaveMemorySEV)
  : _isSaveMemorySEV(isSaveMemorySEV)
{
  initialize(numTax); 		// must be first 
}


TreeAln::TreeAln( const TreeAln& rhs)
  : _taxa(rhs._taxa)
  , _isSaveMemorySEV(rhs._isSaveMemorySEV)
{
  initialize(rhs.getNumberOfTaxa()); // must be first 
  
  auto partCopy = rhs._partitions; 
  setPartitions(partCopy, false); 
  auto blsCpy = rhs._bls; 
  setBranchLengthResource(blsCpy); 

  copyTopologyAndBl(rhs);
}


TreeAln& TreeAln::operator=(TreeAln rhs)
{
  assert(rhs._isSaveMemorySEV == _isSaveMemorySEV); 
  swap(*this, rhs);
  return *this; 
}


void swap(TreeAln& lhs, TreeAln& rhs )
{
  using std::swap; 

  swap(lhs._partitions, rhs._partitions); 
  swap(lhs._partitionListResource, rhs._partitionListResource);
  swap(lhs._partitionPtrs , rhs._partitionPtrs );

  swap(lhs._tr, rhs._tr); 
  swap(lhs._taxa, rhs._taxa); 
  swap(lhs._bls, rhs._bls); 

  swap(lhs._traversalInfo, rhs._traversalInfo); 
  swap(lhs._nodes, rhs._nodes); 
  swap(lhs._nodeptrs, rhs._nodeptrs); 

  swap(lhs._ti, rhs._ti); 
  swap(lhs._execModel, rhs._execModel); 
  swap(lhs._parsimonyScore, rhs._parsimonyScore	);

  swap(lhs._isSaveMemorySEV, rhs._isSaveMemorySEV) ; 
}


void TreeAln::initialize(nat numTax)
{
  memset(&_tr,0,sizeof(pllInstance)); 

  _tr.maxCategories = Partition::maxCategories; 

  _tr.useRecom = PLL_FALSE; 
  _tr.bigCutoff = PLL_FALSE;
  _tr.likelihood =  0 ; 
  _tr.doCutoff = PLL_TRUE;
  _tr.secondaryStructureModel = PLL_SEC_16; /* default setting */ 
  _tr.searchConvergenceCriterion = PLL_FALSE;
  _tr.rateHetModel = PLL_GAMMA; 
  _tr.multiStateModel  = PLL_GTR_MULTI_STATE;

  _tr.manyPartitions = PLL_FALSE;

  _tr.saveMemory = _isSaveMemorySEV ?  PLL_TRUE : PLL_FALSE; 
  _tr.categories = 25;
  _tr.grouped = PLL_FALSE;
  _tr.constrained = PLL_FALSE;
  _tr.gapyness = 0.0; 
  _tr.useMedian = PLL_FALSE;
  _tr.mxtips = numTax; 

  _tr.fastParsimony = PLL_FALSE; 
  _tr.fastScaling = PLL_TRUE; 

  _tr.getParsimonyPerPartition = PLL_TRUE; 

  int tips  = _tr.mxtips;
  int inter = tips -1 ; 
  
  // initialize the traversal info 
  _traversalInfo.resize(tips);
  _tr.td[0].ti = _traversalInfo.data();
  _tr.td[0].count = 0;

  // initialize nodes 
  _nodes.resize(tips + 3 * inter); 

  _nodeptrs.resize(2 * tips, nullptr);
  
  // prepare tip nodes
  auto p0 = begin(_nodes); 
  for (int i = 1; i <= tips; i++)
    {
      auto p = &(*p0); 
      ++p0; 
      p->hash = std::hash<int>()(i); // necessary?
      p->x      =  0;
      p->xBips  = 0;
      p->number =  i;
      p->next   =  p;
      p->back   = (node *)NULL;
      _nodeptrs.at(i) = p; 
    }

  
  // prepare inner nodes 
  for (int i = tips + 1; i <= tips + inter; i++)
    {
      auto p = (node*)nullptr; 
      auto q = (node *)nullptr;
      for (int j = 1; j <= 3; j++)
	{	 
	  p = &(*p0); 
	  ++p0;
	  if(j == 1)
	    {
	      p->xBips = 1;
	      p->x = 1;
	      p->xPars = 1; 
	    }
	  else
	    {
	      p->xBips = 0;
	      p->x =  0;
	      p->xPars = 0; 
	    }
	  p->number = i;
	  p->next   = q;
	  p->back   = (node *) NULL;
	  p->hash   = 0;       
	  q = p;
	}
      p->next->next->next = p;
      _nodeptrs.at(i) = p; 
    }

  _tr.nodep = _nodeptrs.data();

  _tr.likelihood  = PLL_UNLIKELY;
  _tr.ntips       = 0;
  _tr.nextnode    = 0;

  _ti.resize( 4 * getNumberOfTaxa() );
  _tr.ti = _ti.data();
  
  createCaterpillar();

  _tr.start = _tr.nodep[1];
  
  _parsimonyScore.resize( 2 * numTax);
  _tr.parsimonyScore = _parsimonyScore.data();
}


void TreeAln::createCaterpillar()
{
  nat outerCtr = getNumberOfTaxa() +1 ; 
  clipNode(getUnhookedNode(1), getUnhookedNode(outerCtr)); 
  clipNode(getUnhookedNode(2), getUnhookedNode(outerCtr)); 
  nat root = outerCtr; 
  ++outerCtr; 
  for(nat i = 3; i < getNumberOfTaxa() +1; ++i)
    {
      root = addNodeToPartialTree(i,root, outerCtr); 
      // tout << std::endl; 
      ++outerCtr; 
    }
}


void TreeAln::copyTopologyAndBl(const TreeAln &rhs) 
{
  assert(&rhs != this); 
  unlinkTree();

  for(auto &bl : rhs.extractBranches())
    {
      auto a = getUnhookedNode(bl.getPrimNode());
      auto b = getUnhookedNode(bl.getSecNode()); 
      auto z = bl.findNodePtr(rhs)->z; 
      clipNode(a,b, z); 
    }
}


nat TreeAln::addNodeToPartialTree(nat id, nat curRoot, nat outerCtr)
{
  auto tip = getUnhookedNode(id); 
  auto oldRoot = getUnhookedNode(curRoot); 

  if( outerCtr < getNumberOfNodes() +1 )
    {
      clipNode(tip, getUnhookedNode(outerCtr)); 
      clipNode(oldRoot, getUnhookedNode(outerCtr)); 
      return outerCtr; 
    }
  else 
    {
      clipNode(tip, oldRoot); 
      return curRoot; 
    }

}


void TreeAln::clearMemory(ArrayReservoir &arrayReservoir)
{
  for(nat i = 0; i < getNumberOfPartitions() ; ++i)
    {
      auto& partition = getPartition(i).getHandle() ; // TODO: no partition handle  
      for(nat j = 0; j < getNumberOfTaxa() ; ++j)
	{
	  if(partition.xSpaceVector[j] != 0 )
	    {
	      if(partition.xVector[j])
		{
		  arrayReservoir.deallocate(partition.xVector[j]); 
		  partition.xVector[j] = nullptr; 
		  partition.xSpaceVector[j] = 0; 
		}
	    }
	}
    }
}



/** 
 * prunes subtree (primNode is the very root)
 * 
 * returns the original branch of prunedBranch and the branch that we
 * obtain after pruning. This branch after pruning is oriented,
 * s.t. the associated subtree can be traversed correctly.
 */ 
auto  TreeAln::pruneSubtree(const BranchPlain &subtree, const BranchPlain &prunedBranch, const std::vector<AbstractParameter*> &params)
  -> std::tuple<BranchLengths, BranchPlain>
{
  auto result = getBranch(prunedBranch, params);

  auto third = prunedBranch.getThirdBranch(*this,subtree); 
  auto thirdBL =  getBranch(third, params); 

  auto desc = getDescendents(subtree);
  clipNode(desc.first.getInverted().findNodePtr(*this), desc.second.getInverted().findNodePtr(*this)); 

  auto aNode = desc.first.getSecNode(); 
  auto bNode = desc.second.getSecNode(); 
    
  auto newBranch = BranchLengths(aNode, bNode);

  // orientation os the new branch is important. Since we take away
  // prunedBranch, the other node that is not in prunedBranch must be
  // at the very root
  if( prunedBranch.hasNode(aNode) )
    newBranch = newBranch.getInverted();

  newBranch.setLengths(thirdBL.getLengths()); 
  setBranch(newBranch, params); 

  // cut the subtree  
  detachNode(subtree.findNodePtr(*this)); 

  tout << "PRUNE: " << SHOW(subtree) << SHOW(result) << SHOW(newBranch.toPlain()) << std::endl; 

  return std::make_tuple(result, newBranch.toPlain()); 
} 


/** 
    inserts a previously pruned subtree, correctly adding the novel branch  
    
    assumes that subtree is in a pruned state  
  
    note that prunedBranch must be a branch that will exist after insertion
 */ 
void TreeAln::insertSubtree(const BranchPlain &subtree, const BranchPlain& insertBranch, const BranchLengths &branchToCreate, const std::vector<AbstractParameter*> &params)
{
  // need to dance around the sensitive findNodePtr
  auto pBack = subtree.getInverted().findNodePtr(*this); 
  auto p = pBack->back; 
  assert(p->next->back == NULL  &&   p->next->next->back == NULL); 

  auto otherBranchToCreate = getBranch(insertBranch, params); 
  otherBranchToCreate.setPrimNode(subtree.getPrimNode()); 
  otherBranchToCreate.setSecNode(  insertBranch.getOtherNode(branchToCreate.getOtherNode(subtree.getPrimNode()))  );
  
  // cut the insert branch 
  auto q = insertBranch.findNodePtr(*this); 
  q->back = NULL; 
  auto r = insertBranch.getInverted().findNodePtr(*this); 
  r->back = NULL; 

  clipNode(p->next , q); 
  clipNode(p->next->next, r ); 
  
  setBranch(branchToCreate, params); 
  setBranch(otherBranchToCreate, params); 

  tout << "INSERT: " << SHOW(subtree) << SHOW(insertBranch) << SHOW(branchToCreate) << std::endl; 
}



void TreeAln::clipNodeDefault(nodeptr p, nodeptr q )
{
  clipNode(p,q);  
  // use a problematic branch length to ensure, that it was overridden
  for(nat i = 0; i < getNumberOfPartitions(); ++i)
    p->z[i] = p->back->z[i] = 0.9; 
}


void TreeAln::clipNode(nodeptr p, nodeptr q, double *z )
{
  p->back = q; 
  q->back = p; 

  if(z != nullptr)
    {
      for(nat i = 0; i < getNumberOfPartitions(); ++i)
	p->z[i] = q->z[i] = z[i]; 
    }
}



void TreeAln::detachNode(nodeptr p)
{
  p->next->back = NULL; 
  p->next->next->back = NULL; 
}

void TreeAln::unlinkNode(nodeptr p )
{
  p->back = NULL; 
  detachNode(p); 
}


void TreeAln::unlinkTree()
{
#ifdef DEBUG_LINK_INFO
  cout << "unlinking everything" << endl; 
#endif

  // unlink tips 
  for(int i = 1; i < _tr.mxtips+1; ++i)
    {
      nodeptr p = _tr.nodep[i]; 
      p->back = NULL; 
    }

  for(int i = _tr.mxtips + 1; i < 2 * _tr.mxtips; ++i)
    unlinkNode(_tr.nodep[i]); 
}


nodeptr TreeAln::getUnhookedNode(int number)
{  
  nodeptr p = _tr.nodep[number]; 

  if(isTipNode(number) )
    {
      assert(p->back == NULL); 
      return p; 
    }

  nodeptr q = p ; 
  do 
    {
      if(q->back == NULL)
	return q ; 
      q = q->next; 
    } while(p != q); 

  
  std::cerr << "Error: did not find unlinked node for " << number << std::endl; 
  assert(0);
  return NULL;
}


nat TreeAln::getNumberOfPartitions() const
{
  return _partitions.size(); 
}


const Partition& TreeAln::getPartition(nat model)  const 
{
  return _partitions.at(model);
}

Partition& TreeAln::getPartition(nat model)   
{
  return _partitions.at(model);
}


std::vector<bool> TreeAln::getExecModel() const 
{ 
  auto result = std::vector<bool>{}; 

  for(auto &p : _partitions)
    result.push_back(p.getHandle().executeModel);

  return result; 
}
 
void TreeAln::setExecModel(const std::vector<bool>  &modelInfo)
{
  nat ctr = 0; 
  for(auto &p : _partitions)
    {
      p.getHandle().executeModel =  modelInfo[ctr] ?  true : false; 
      ++ctr;
    }
}

std::vector<log_double> TreeAln::getPartitionLnls() const
{
  auto result = std::vector<log_double>{}; 

  for(auto &p : _partitions)
    result.push_back(   log_double::fromLog(p.getHandle().partitionLH)  ); 

  return result; 
}



void TreeAln::setPartitionLnls(const std::vector<log_double> partitionLnls)  
{
  nat ctr = 0; 
  for(auto &p : _partitions )
    {
      p.getHandle().partitionLH = partitionLnls[ctr].getRawLog(); 
      ++ctr; 
    }
} 


void TreeAln::initRevMat(nat model)
{
  initReversibleGTR(&getTrHandle(), &(getPartitionsHandle()) , model); 
}


void TreeAln::setRevMat(const std::vector<double> &values, nat model, bool isRaw )
{
  bool valuesOkay = BoundsChecker::checkRevmat(values); 
  if(not valuesOkay)
    {
      tout << "Problem with substitution parameter: "  << MAX_SCI_PRECISION << values << std::endl;  
      assert( valuesOkay ); 
    }

  auto& partition = getPartition(model) ; 

  assert(partition.getDataType() != PLL_AA_DATA || partition.getProtModels() == PLL_GTR); 

  nat num = RateHelper::numStateToNumInTriangleMatrix(  partition.getStates()); 
  assert(num == values.size()); 

  partition.setSubstRates(values); 

  initRevMat(model); 
}


void TreeAln::setBranch(const BranchLength& branch, const AbstractParameter* param)
{
  // tout << "setting " << branch << std::endl; 

  assert(BoundsChecker::checkBranch(branch)); 
  assert(branch.exists(*this)); 

  auto p = branch.findNodePtr(*this);
  for(auto &partition : param->getPartitions() )
    {
      double length = branch.getLength(); 
      p->z[partition] = p->back->z[partition] = length; 
    }
}


void TreeAln::setBranch(const BranchLengths& branch, const std::vector<AbstractParameter*>params)
{
  assert(BoundsChecker::checkBranch(branch)); 
  assert(branch.exists(*this)); 

  auto p = branch.findNodePtr(*this); 
  
  for(auto &param : params)
    {
      double length = branch.getLengths().at(param->getIdOfMyKind());
      for(auto &partition : param->getPartitions())
	p->z[partition] = p->back->z[partition] = length; 
    }
}


void TreeAln::setAlpha(double alpha,  nat model)
{
  assert(BoundsChecker::checkAlpha(alpha)); 
  auto& p = getPartition(model); 
  p.setAlpha(alpha); 
  discretizeGamma(model); 
}


/**
   @brief makes the discrete categories for the gamma
   distribution. Has to be called, if alpha was updated.   

*/ 
void TreeAln::discretizeGamma(nat model)
{
  auto& partition =  getPartition(model).getHandle(); 
  makeGammaCats(partition.alpha, partition.gammaRates, Partition::maxCategories, _tr.useMedian);
}


std::ostream& operator<< (std::ostream& out,  const TreeAln&  traln)
{
  auto tp = TreePrinter(true, false, false); 
  return out << tp.printTree(traln); 
}


std::vector<double> TreeAln::getRevMat(nat model, bool isRaw) const 
{
  auto& partition = getPartition(model); 
  assert(partition.getDataType() != PLL_AA_DATA || partition.getProtModels() == PLL_GTR); 

  auto result = partition.getSubstRates();

  if(not isRaw)
    RateHelper::convertToSum1(result);

  return result; 
}


void TreeAln::setFrequencies(const std::vector<double> &values, nat model)
{
  assert( BoundsChecker::checkFrequencies(values) ) ;    
  auto& partition = getPartition(model); 
  assert( partition.getDataType() != PLL_AA_DATA || partition.getProtFreqs() == PLL_TRUE ); 

  partition.setFrequencies(values); 
  initRevMat(model);   
}


std::vector<double> TreeAln::getFrequencies(nat model) const
{
  auto& partition = getPartition(model) ; 
  assert( partition.getDataType() != PLL_AA_DATA || partition.getProtFreqs() == PLL_TRUE ); 
  return partition.getFrequencies();
}


nodeptr TreeAln::getNode(nat elem) const  
{
  if( not (elem != 0 && elem < getNumberOfNodes() + 2 ))
    { 
      std::cout << "bug: attempted to get node " << elem << std::endl; assert(elem != 0 && elem <= getNumberOfNodes() + 1 ) ;
    } 

  return  getTrHandle().nodep[elem] ; 
}


std::pair<BranchPlain,BranchPlain> TreeAln::getDescendents(const BranchPlain &b) const
{
  auto p = b.findNodePtr(*this); 
  
  assert(not isTipNode(p)); 

  auto pn = p->next,
    pnn = p->next->next; 
  return 
    std::make_pair ( BranchPlain(pn->number, pn->back->number), 
		     BranchPlain(pnn->number, pnn->back->number) ); 
} 


double TreeAln::getMeanSubstitutionRate(const std::vector<nat> &partitions) const 
{
  double result = 0; 
  auto sum = double{0.}; 

  for(auto &p :partitions)
    {
      auto& partition =  getPartition(p);
      result += partition.getFracChange()  * partition.getPartitionContribution();  
      sum += partition.getPartitionContribution(); 
      assert(partition.getPartitionContribution() > 0 ) ; 
      assert( partition.getFracChange() > 0  ); 
    }

  return result / sum   ; 
}


nat TreeAln::getDepth(const BranchPlain &b) const 
{
  if(b.isTipBranch(*this))
    return 1; 
  else 
    {
      auto desc = getDescendents(b.getInverted());
      return std::max(getDepth(desc.first), getDepth(desc.second)) + 1; 
    }
}


std::vector<nat> TreeAln::getLongestPathBelowBranch(const BranchPlain &b) const 
{
  if(b.isTipBranch(*this))
    return std::vector<nat>{b.getSecNode()}; 
  else 
    {
      auto desc = getDescendents(b.getInverted());
      auto resA = getLongestPathBelowBranch(desc.first);
      auto resB = getLongestPathBelowBranch(desc.second);
      if(resA.size() > resB.size())
	{
	  resA.push_back(b.getSecNode()); 
	  return resA; 
	}
      else 
	{
	  resB.push_back(b.getSecNode()); 
	  return resB; 
	}
    }
}


std::vector<nat> TreeAln::getNeighborsOfNode( nat node ) const 
{
  auto result = std::vector<nat>{};
  auto nodePtr = getNode(node); 
  if(isTipNode(nodePtr))
    result.push_back(nodePtr->back->number);
  else 
    {
      result = 
	{
	  nat(nodePtr->back->number), 
	  nat(nodePtr->next->back->number), 
	  nat(nodePtr->next->next->back->number)
	} ; 
    }
  
  return result; 
} 


BranchPlain TreeAln::getAnyBranch() const 
{
  // assert(0); 
  return BranchPlain(_tr.nodep[1]->number, _tr.nodep[1]->back->number); 
} 

BranchLength TreeAln::getBranch(const nodeptr &p,  const AbstractParameter *param) const
{
  return getBranch(BranchPlain(p->number,p->back->number), param); 
}


BranchLengths TreeAln::getBranch(const nodeptr& p, const std::vector<AbstractParameter*> &params) const 
{
  return getBranch(BranchPlain(p->number, p->back->number), params); 
}


BranchLength TreeAln::getBranch(const BranchPlain &branch,  const AbstractParameter *param) const
{
  auto result = branch.toBlDummy();
  result.extractLength(*this, branch, param); 
  return result; 
}


BranchLengths TreeAln::getBranch(const BranchPlain& branch, const std::vector<AbstractParameter*> &params) const 
{
  auto result = branch.toBlsDummy();
  result.extractLength(*this, branch, params);
  return result; 
}


nat TreeAln::getNumberOfAssignedSites(nat model) const 
{
  auto partition =  getPartition (model); 
  nat length = partition.getWidth(); 
  return length; 
}


std::vector<BranchPlain> TreeAln::extractBranches() const
{
  auto result = std::vector<BranchPlain>(); 
  nat numBranch = getNumberOfBranches(); 
  result.reserve(numBranch); 

  int last = int(getNumberOfNodes() + 1); 

  for(int i = getNumberOfTaxa() + 1 ; i < last ; ++i)
    {
      auto node = getNode(i); 
      
      if( node->back->number < i )
	result.emplace_back(i,node->back->number); 
      if(node->next->back->number < i)
	result.emplace_back(i,node->next->back->number); 
      if(node->next->next->back->number < i)
	result.emplace_back(i,node->next->next->back->number); 
    }

  assert(result.size() == getNumberOfBranches());

  return result;
}


std::vector<BranchLength> TreeAln::extractBranches( const AbstractParameter* param ) const 
{
  auto result = std::vector<BranchLength>(); 
  result.reserve(getNumberOfBranches());

  for(int i = getNumberOfTaxa() + 1 ; i < int(getNumberOfNodes() + 1 )   ; ++i)
    {
      auto node = getNode(i); 
      
      if( node->back->number < i )
	result.emplace_back( getBranch(BranchPlain(   i,node->back->number ),param) ); 
      if(node->next->back->number < i)
	result.emplace_back( getBranch(BranchPlain(i,node->next->back->number), param) ); 
      if(node->next->next->back->number < i)
	result.emplace_back( getBranch(BranchPlain( i,node->next->next->back->number ),param) );  
   }

  assert(result.size() == getNumberOfBranches());
  return result;
}

std::vector<BranchLengths> TreeAln::extractBranches( const std::vector<AbstractParameter*> &param ) const 
{
  auto result = std::vector<BranchLengths>(); 
  result.reserve(getNumberOfBranches()); 

  for(int i = getNumberOfTaxa() + 1 ; i < int(getNumberOfNodes() + 1 )  ; ++i)
    {
      auto node = getNode(i); 
      
      if( node->back->number < i )
	result.emplace_back(  getBranch(BranchPlain(   i,node->back->number ),param) ); 
      if(node->next->back->number < i)
	result.emplace_back(  getBranch(BranchPlain(i,node->next->back->number), param)); 
      if(node->next->next->back->number < i)
	result.emplace_back( getBranch(BranchPlain( i,node->next->next->back->number ),param) ); 
    }

  assert(result.size() == getNumberOfBranches());
  return result;
}


std::vector<BranchPlain> TreeAln::getBranchesByDistance(const BranchPlain& branch, nat distance, bool bothSides ) const 
{
  if(distance == 0)
    return { branch };

  auto toCheck = std::vector<BranchPlain>{}; 
  
  if(not isTipNode(branch.findNodePtr(*this))) 
    {
      auto desc = getDescendents(branch); 
      toCheck.push_back(desc.first.getInverted()); 
      toCheck.push_back(desc.second.getInverted()); 
    }

  if(bothSides
     && not isTipNode(branch.getInverted().findNodePtr(*this)))
    {
      auto desc2 = getDescendents(branch.getInverted()); 
      toCheck.push_back(desc2.first.getInverted()); 
      toCheck.push_back(desc2.second.getInverted()); 
    }

  auto myResult = std::vector<BranchPlain>{}; 
  for(auto b : toCheck)
    {
      auto result = getBranchesByDistance(b, distance-1, false); 
      myResult.insert(end(myResult), begin(result), end(result)); 
    }

  return myResult; 
}


void TreeAln::setProteinModel(int part, ProtModel model) 
{
  auto& partition = getPartition(part) ; 
  partition.setProtModel(int(model)); 
  initRevMat(part); 
}


ProtModel TreeAln::getProteinModel(int part) const
{
  auto& pData = getPartition(part) ; 
  return ProtModel(pData.getProtModels()); 
}


// HACK, don't use 
void TreeAln::setBranchUnchecked(const BranchLength &bl)
{
  auto p =  bl.findNodePtr(*this);
  p->z[0] = bl.getLength();
  p->back->z[0] = bl.getLength();
}


void TreeAln::clearMemory()
{
  // TODO redundant with allocator function!

  for(nat i = 0; i < getNumberOfPartitions() ; ++i )
    {
      auto &partition = getPartition(i).getHandle();
      for(nat j = 0; j < getNumberOfTaxa(); ++j)
	if(partition.xSpaceVector[j] != 0 )
	  {
	    free(partition.xVector[j] );
	    partition.xSpaceVector[j] = 0; 
	  }
    }
} 


void TreeAln::setBranchLengthResource(BranchLengthResource bls)
{
  _bls = bls; 
  _bls.assign(*this);
}


partitionList& TreeAln::getPartitionsHandle() 
{ 
  return _partitionListResource; 
}
   
const partitionList& TreeAln::getPartitionsHandle() const
{ 
  return _partitionListResource; 
}


// TODO not entirely happy with the initial part 
void TreeAln::setPartitions(const std::vector<Partition> &p, bool initial)
{
  _partitions = p;
  _partitionPtrs.resize(0);
  for(auto & part : _partitions)
    _partitionPtrs.push_back(&(part.getHandle())); 
  _partitionListResource.partitionData = _partitionPtrs.data(); 
  _partitionListResource.perGeneBranchLengths = PLL_TRUE; 
  _partitionListResource.numberOfPartitions = _partitions.size(); 
  _partitionListResource.dirty = PLL_FALSE; 

  if(initial)
    {
      auto empFreqs = std::vector<double*>(getNumberOfPartitions()); 
      for(nat i = 0; i < getNumberOfPartitions() ; ++i)
  	{
  	  auto& partition = getPartition(i); 
  	  empFreqs[i] = new double[partition.getStates()]; 
  	  for(int j = 0; j < partition.getStates() ; ++j)
  	    empFreqs[i][j] = 1. / partition.getStates(); 
  	}
      initModel(&(getTrHandle()), empFreqs.data(), &(getPartitionsHandle()));

      for(nat i = 0; i < getNumberOfPartitions(); ++i)
  	delete [] empFreqs[i];
    }

  _execModel.resize(p.size(), PLL_TRUE);
  _tr.td[0].executeModel = _execModel.data();
}




