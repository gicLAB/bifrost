
// JSONCPP
#include "json/json.h"
#include "json/json-forwards.h"

// File handling
#include <fstream>
#include <string>
#include <iostream>

// This creates the JSON file used for tuning

void reportCost(
    std::string tuning_name,
    std::string filename,
    int cost)
{
    // Intialise the JSONCPP variables
    Json::Value root;
    Json::Reader reader;
    Json::StyledStreamWriter writer;

    // Read the file
    std::ifstream f(filename);

    // Parse the file
    bool parsingSuccessful = reader.parse(f, root);
    if (!parsingSuccessful)
    {
        // report to the user the failure and their locations in the document.
        std::cout << "Failed to parse configuration\n"
                    << reader.getFormattedErrorMessages();
        return;
    }
    f.close();
    // Add in the recorded cost
    if (root["tuning_name"] == tuning_name)
    {
        root["value"].append(cost);
    }
    else
    {
        // Create new member and insert array with one value
        Json::Value content(Json::arrayValue);
        content.append(cost);
        root["value"] = content;

        // Change tuning name variable
        root["tuning_name"] = tuning_name;
    }
    // Write output
    std::ofstream fout(filename);
    writer.write(fout, root);
}

void reportTotalCycles(
    std::string tuning_name,
    std::string filename,
    int cycles)
{
    // Intialise the JSONCPP variables
    Json::Value root;
    Json::Reader reader;
    Json::StyledStreamWriter writer;

    // Read the file
    std::ifstream f(filename);

    // Parse the file
    bool parsingSuccessful = reader.parse(f, root);
    if (!parsingSuccessful)
    {
        // report to the user the failure and their locations in the document.
        std::cout << "Failed to parse configuration\n"
                    << reader.getFormattedErrorMessages();
        return;
    }
    f.close();
     
    root["value"].append(cycles);
    root["layer"].append(tuning_name);

    // Write output
    std::ofstream fout(filename);
    writer.write(fout, root);
}