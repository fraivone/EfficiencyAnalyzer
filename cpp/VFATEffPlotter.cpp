#include "VFATEffPlotter.h"

std::string chamberName(int station, int region, int chamber, int layer, int chamberType){

    std::string str_region;
	if (region == 1) str_region = "P";
	else str_region = "M";
	std::stringstream ss;
    ss << "GE" << station << "1-"<<str_region<<"-"<<std::setfill('0')<<std::setw(2)<<chamber<<"L"<<layer;
	return ss.str();
}

std::string unitName(int station, int region, int chamber, int layer, int chamberType){
    if(station == 2)
        return chamberName(station, region, chamber, layer, chamberType)+letter[chamberType];
    return chamberName(station, region, chamber, layer, chamberType);

}

void SetPalette(){
// ### Setting up palette
UInt_t NRGBs = 5;
UInt_t NContours = 99;
Double_t red[5] =     { 0., 1.00, 1.00, 0.51, 0.00 };
Double_t green[5] =   { 0., 0.00, 1.00, 1.00, 1.00 };
Double_t blue[5] =    { 0., 0.00, 0.00, 0.00, 0.00 };
Double_t stops[5] =   { 0., 0.02, 0.65, 0.95, 1.00 };
Int_t palette_number = TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, NContours);
gStyle->SetNumberContours(NContours);
}

/*Builds efficiency plots 1D and 2D for the GE11 and GE21 chambers listed in the csv input file
the output is stored in the parsed folder.

GE11 and GE21 are treated a bit differently:
GE11:
    1 unit (chamber) per plot 2D, 24 VFATs
    1 unit (chamber) per plot 1D, 24 VFATs
    EfficiencyCollector[unit_name] = (VFAT Number, Matched, Propagated)    
GE21:
    4 units (1 chamber) per plot 2D, 12x4 = 48 VFATs
    4 units (1 chamber) per plot 1D, 4 x (12VFAT plots)
    EfficiencyCollector[chamber name (not nodule)] = (VFAT Number + (modulenumber-1)*12, Matched, Propagated)

 */
int main(int argc, char* argv[]) {
 
	 if (argc != 3){
		std::cout<<"Exepcted arguments: <csv_path> <output_folder_name>"<<std::endl;
        std::cout<<"\nExiting\n\n"<<std::endl;
		return 0;
	 }

     char* indexPath = std::getenv("INDEXPHP");         
     
	std::string csv_path = argv[1];

    std::filesystem::path output_folder_name = std::filesystem::absolute(std::string(argv[2]));
    std::filesystem::create_directories(output_folder_name);
    std::filesystem::create_directories(output_folder_name / "VFAT");

    if (indexPath != nullptr){
        std::string strPath(indexPath);
        std::filesystem::path phpPath = strPath;

        std::filesystem::copy_options options = std::filesystem::copy_options::overwrite_existing;
        std::filesystem::copy(phpPath, output_folder_name / phpPath.filename(), options);
        output_folder_name /= "VFAT/";
        std::filesystem::copy(phpPath, output_folder_name / phpPath.filename(), options);
    }
    else
        output_folder_name /= "VFAT/";

    gROOT->SetBatch(1);
    TCanvas *theCanvas = new TCanvas("c1","c1",1080,1080);
    TCanvas *theCanvas4Pads = new TCanvas("c4Pads","c4Pads",1080,4320);
    theCanvas4Pads->Divide(1,4,0.01,0.001);
	SetPalette();

    auto TH2Short = GetTH2Poly(1, 2);
    auto TH2Long = GetTH2Poly(1, 2);
    auto TH2GE21 = GetTH2Poly(2, 1);

    GE11VfatPlot::TH1GE11 = GetTH1Plot();
    for(int k = 0; k<GE21VfatPlot::N_GE21MODULES; k++){
        GE21VfatPlot::TH1GE21[k] = GetTH1Plot();
    }
    
    
    // Reading file
    std::ifstream infile(argv[1]);
    getline(infile, line,'\n'); // Skip header
	while (getline(infile, line,'\n')){
         row.clear();
         std::stringstream str(line);
        while(getline(str, word, ','))
			row.push_back(word);
		content.push_back(row);
    }

	//int s = content.size();
    for(const auto & item : content) {
        int station = std::stoi(item[0]),  region = std::stoi(item[1]), chamber = std::stoi(item[2]),  layer = std::stoi(item[3]), chamberType = std::stoi(item[4]);
        auto name = chamberName( station, region, chamber, layer, chamberType);
        if (station == 2){
            auto VFAToffset = 12*(chamberType-1);
            EfficiencyCollector[name].emplace_back(std::stoi(item[5]) + VFAToffset, std::stoi(item[6]), std::stoi(item[7]));
        }
        if (station == 1)
            EfficiencyCollector[name].emplace_back(std::stoi(item[5]), std::stoi(item[6]), std::stoi(item[7]));
    }

	// Filling histos
	for (auto const& x : EfficiencyCollector){
		auto this_chamberName = x.first;
		auto this_chamberHits = x.second;
        
        isGE11 = this_chamberName.find("GE11")==std::string::npos ? false : true;
        isGE21 = this_chamberName.find("GE21")==std::string::npos ? false : true;
        if (isGE11)
            isLong = std::stoi(this_chamberName.substr(this_chamberName.length() - 3, 1)) % 2 == 0;
        TH2Poly* Plot2D;
        if (isGE11)
            Plot2D = isLong ? (TH2Poly*)TH2Long->Clone() : (TH2Poly*)TH2Short->Clone();
        if (isGE21)
            Plot2D = (TH2Poly*)TH2GE21->Clone();
		Plot2D->Reset("ICES");

        GE11VfatPlot::reset();
		GE21VfatPlot::reset();
		for(const auto & item : this_chamberHits){
            if (isGE21)
                moduleNumber = item.vfat / 12 + 1;
			if (item.propagatedHits != 0){
				float eff = (float)item.matchedHits/(float)item.propagatedHits;
				eff = round(eff * 1000.0) / 1000.0; // Round to the 3rd decimal
				Plot2D->SetBinContent(item.vfat+1,eff);
			}
            if (isGE11){
                GE11VfatPlot::numerator->SetBinContent(item.vfat+1,item.matchedHits);
                GE11VfatPlot::denominator->SetBinContent(item.vfat+1,item.propagatedHits);
            }
            if(isGE21){
                GE21VfatPlot::numerator[moduleNumber-1]->SetBinContent(item.vfat%12+1,item.matchedHits);
                GE21VfatPlot::denominator[moduleNumber-1]->SetBinContent(item.vfat%12+1,item.propagatedHits);
            }
        }

        theCanvas->cd();
        Plot2D->SetTitle(this_chamberName.c_str());
        Plot2D->SetName(this_chamberName.c_str());
        Plot2D->Draw("COLZ TEXT");
        TH2_output_name = output_folder_name / (this_chamberName+"_Eff2D");
        theCanvas->SaveAs((TH2_output_name.string()+".png").c_str());
		theCanvas->SaveAs((TH2_output_name.string()+".pdf").c_str());
		delete Plot2D;

        if(isGE11){
            GE11VfatPlot::TH1GE11->SetTitle(this_chamberName.c_str());
            GE11VfatPlot::TH1GE11->SetName(this_chamberName.c_str());
            GE11VfatPlot::TH1GE11->Divide(GE11VfatPlot::numerator,GE11VfatPlot::denominator,"B");
            GE11VfatPlot::TH1GE11->GetXaxis()->SetLimits(-0.5,23.5);
            GE11VfatPlot::TH1GE11->GetXaxis()->SetNdivisions(24);

            theCanvas->cd();
            GE11VfatPlot::TH1GE11->Draw("APE");
            theCanvas->SetGrid();
            TH1_output_name = output_folder_name.string()+this_chamberName+"_Eff1D";
            theCanvas->SaveAs((TH1_output_name+".png").c_str());
            theCanvas->SaveAs((TH1_output_name+".pdf").c_str());
        }
        if(isGE21){
            for(int k = 0; k<GE21VfatPlot::N_GE21MODULES; k++){
                title = this_chamberName + letter[k+1];
                GE21VfatPlot::TH1GE21[k]->SetTitle(title.c_str());
                GE21VfatPlot::TH1GE21[k]->SetName(title.c_str());
                GE21VfatPlot::TH1GE21[k]->Divide(GE21VfatPlot::numerator[k],GE21VfatPlot::denominator[k],"B");
                GE21VfatPlot::TH1GE21[k]->GetXaxis()->SetLimits(-0.5,11.5);
                GE21VfatPlot::TH1GE21[k]->GetXaxis()->SetNdivisions(12);
                
                theCanvas4Pads->cd(GE21VfatPlot::N_GE21MODULES-k)->SetGrid();
                GE21VfatPlot::TH1GE21[k]->Draw("APE");
            }
            TH1_output_name = output_folder_name.string()+this_chamberName+"_Eff1D";
            theCanvas4Pads->SaveAs((TH1_output_name+".png").c_str());
            theCanvas4Pads->SaveAs((TH1_output_name+".pdf").c_str());
        }
	}
	
    return 0;
}
