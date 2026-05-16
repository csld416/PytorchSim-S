//Abstract class that represents the distribution network.

#ifndef __DistributionNetworkAbstract__
#define __DistributionNetworkAbstract__

#include <vector>
#include <iostream>

#include "DSNetwork.h"
#include "Connection.h"
#include <map>
#include <iostream>
#include <assert.h>


//This class represents a general case of a distribution network and cannot be instantiated

class DistributionNetwork : public Unit {
private:
  
public:
    //General constructor, just used to heritage with unit
    DistributionNetwork(id_t id, std::string name) : Unit(id, name) {}
    //This just executes cycle over all the dsnetworks
    virtual void cycle() {assert(false);} 
    //Get last levels connections together. Useful to connect with mswitches later.
    virtual std::map<int, Connection*> getLastLevelConnections()=0;
    // Get the top connections (i.e., the ones that connect the SDMemory ports)
    virtual std::vector<Connection*> getTopConnections()=0;
    virtual void printStats(std::ofstream& out, unsigned int indent)=0;
    virtual void printEnergy(std::ofstream& out, unsigned int indent)=0;

};

#endif