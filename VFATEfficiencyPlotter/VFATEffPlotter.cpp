#include "VFATEffPlotter.h"

std::string chamberName(int station, int region, int chamber, int layer){
	std::string str_region;
	
	if (region == 1) str_region = "P";
	else str_region = "M";
	std::stringstream ss;
	ss << "GE" << station << "1-"<<str_region<<"-"<<std::setfill('0')<<std::setw(2)<<chamber<<"L"<<layer;
	
	return ss.str();
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


int main(int argc, char* argv[]) {
	 if (argc != 3){
		std::cout<<"Wrong usage"<<std::endl;
		std::cout<<"Exepcted arguments: <csv_path> <output_folder_name>\nExiting\n\n"<<std::endl;
		return 0;
	 }
	std::string csv_path = argv[1];
	std::string output_folder_name = "/eos/user/f/fivone/www/P5_Operations/Run3/"+std::string(argv[2]);
	system(("mkdir -p "+output_folder_name).c_str());
	system(("cp /eos/user/f/fivone/www/index.php "+output_folder_name).c_str());
	output_folder_name += "/VFAT/";
	system(("mkdir -p "+output_folder_name).c_str());
	system(("cp /eos/user/f/fivone/www/index.php "+output_folder_name).c_str());

    gROOT->SetBatch(1);
	SetPalette();

    auto TH2Short = GE11VfatPlot::GetTH2PolyShort();
    auto TH2Long = GE11VfatPlot::GetTH2PolyLong();  
    auto TH1Plot = GE11VfatPlot::GetTH1Plot();
    
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
    
	int s = content.size();    
    for(const auto & item : content) {
	    auto name = chamberName(
			std::stoi(item[0]),
			std::stoi(item[1]),
			std::stoi(item[2]),
			std::stoi(item[3]));
			
        EfficiencyCollector[name].emplace_back(std::stoi(item[4]), std::stoi(item[5]), std::stoi(item[6]));
    }
    
	// Filling histos
	for (auto const& x : EfficiencyCollector){
		auto this_chamberName = x.first;
		auto this_chamberHits = x.second;
		bool isLong = std::stoi(this_chamberName.substr(this_chamberName.length() - 3, 1)) % 2 == 0;
		TH2Poly* Plot2D = isLong ? (TH2Poly*)TH2Long->Clone() : (TH2Poly*)TH2Short->Clone();
		Plot2D->Reset("ICES");
		GE11VfatPlot::numerator->Reset("ICES");
		GE11VfatPlot::denominator->Reset("ICES");
		for(const auto & item : this_chamberHits){
			if (item.propagatedHits != 0){
				float eff = (float)item.matchedHits/(float)item.propagatedHits;
				eff = round(eff * 1000.0) / 1000.0; // Round to the 3rd decimal
				Plot2D->SetBinContent(item.vfat+1,eff);
			}
        	GE11VfatPlot::numerator->SetBinContent(item.vfat+1,item.matchedHits);
        	GE11VfatPlot::denominator->SetBinContent(item.vfat+1,item.propagatedHits);
		}
		
        Plot2D->SetTitle(this_chamberName.c_str());
        Plot2D->SetName(this_chamberName.c_str());
		Plot2D->Draw("COLZ TEXT");
		TH2_output_name = output_folder_name+this_chamberName+"_Eff2D";
		GE11VfatPlot::c->SaveAs((TH2_output_name+".png").c_str());
		GE11VfatPlot::c->SaveAs((TH2_output_name+".pdf").c_str());
		delete Plot2D;

		TH1Plot->SetTitle(this_chamberName.c_str());
        TH1Plot->SetName(this_chamberName.c_str());
		TH1Plot->Divide(GE11VfatPlot::numerator,GE11VfatPlot::denominator,"B");
    	TH1Plot->GetXaxis()->SetLimits(-0.5,23.5);
    	TH1Plot->GetXaxis()->SetNdivisions(24);
    	TH1Plot->Draw("APE");
    	GE11VfatPlot::c->SetGrid();
		TH1_output_name = output_folder_name+this_chamberName+"_Eff1D";
    	GE11VfatPlot::c->SaveAs((TH1_output_name+".png").c_str());
    	GE11VfatPlot::c->SaveAs((TH1_output_name+".pdf").c_str());
	}
	
    return 0;
}