/**
 * Country name to ISO 3166-1 alpha-2 code mapping
 * Used for circle-flags integration
 */

const COUNTRY_TO_ISO: Record<string, string> = {
  // Europe
  "Albania": "al",
  "Andorra": "ad",
  "Armenia": "am",
  "Austria": "at",
  "Azerbaijan": "az",
  "Belarus": "by",
  "Belgium": "be",
  "Bosnia": "ba",
  "Bosnia-Herzegovina": "ba",
  "Bosnia and Herzegovina": "ba",
  "Bulgaria": "bg",
  "Croatia": "hr",
  "Cyprus": "cy",
  "Czech-Republic": "cz",
  "Czech Republic": "cz",
  "Denmark": "dk",
  "England": "gb-eng",
  "Estonia": "ee",
  "Faroe-Islands": "fo",
  "Faroe Islands": "fo",
  "Finland": "fi",
  "France": "fr",
  "Georgia": "ge",
  "Germany": "de",
  "Gibraltar": "gi",
  "Greece": "gr",
  "Hungary": "hu",
  "Iceland": "is",
  "Ireland": "ie",
  "Israel": "il",
  "Italy": "it",
  "Kazakhstan": "kz",
  "Kosovo": "xk",
  "Latvia": "lv",
  "Liechtenstein": "li",
  "Lithuania": "lt",
  "Luxembourg": "lu",
  "Malta": "mt",
  "Moldova": "md",
  "Monaco": "mc",
  "Montenegro": "me",
  "Netherlands": "nl",
  "North-Macedonia": "mk",
  "North Macedonia": "mk",
  "Northern-Ireland": "gb-nir",
  "Northern Ireland": "gb-nir",
  "Norway": "no",
  "Poland": "pl",
  "Portugal": "pt",
  "Romania": "ro",
  "Russia": "ru",
  "San-Marino": "sm",
  "San Marino": "sm",
  "Scotland": "gb-sct",
  "Serbia": "rs",
  "Slovakia": "sk",
  "Slovenia": "si",
  "Spain": "es",
  "Sweden": "se",
  "Switzerland": "ch",
  "Turkey": "tr",
  "Ukraine": "ua",
  "Wales": "gb-wls",

  // South America
  "Argentina": "ar",
  "Bolivia": "bo",
  "Brazil": "br",
  "Chile": "cl",
  "Colombia": "co",
  "Ecuador": "ec",
  "Paraguay": "py",
  "Peru": "pe",
  "Uruguay": "uy",
  "Venezuela": "ve",

  // North/Central America & Caribbean
  "Canada": "ca",
  "Costa-Rica": "cr",
  "Costa Rica": "cr",
  "Cuba": "cu",
  "Dominican-Republic": "do",
  "Dominican Republic": "do",
  "El-Salvador": "sv",
  "El Salvador": "sv",
  "Guatemala": "gt",
  "Haiti": "ht",
  "Honduras": "hn",
  "Jamaica": "jm",
  "Mexico": "mx",
  "Nicaragua": "ni",
  "Panama": "pa",
  "Puerto-Rico": "pr",
  "Puerto Rico": "pr",
  "Trinidad-And-Tobago": "tt",
  "Trinidad and Tobago": "tt",
  "USA": "us",
  "United-States": "us",
  "United States": "us",

  // Asia
  "Australia": "au",
  "Bahrain": "bh",
  "Bangladesh": "bd",
  "China": "cn",
  "Hong-Kong": "hk",
  "Hong Kong": "hk",
  "India": "in",
  "Indonesia": "id",
  "Iran": "ir",
  "Iraq": "iq",
  "Japan": "jp",
  "Jordan": "jo",
  "Kuwait": "kw",
  "Kyrgyzstan": "kg",
  "Lebanon": "lb",
  "Malaysia": "my",
  "Oman": "om",
  "Pakistan": "pk",
  "Palestine": "ps",
  "Philippines": "ph",
  "Qatar": "qa",
  "Saudi-Arabia": "sa",
  "Saudi Arabia": "sa",
  "Singapore": "sg",
  "South-Korea": "kr",
  "South Korea": "kr",
  "Korea": "kr",
  "Syria": "sy",
  "Taiwan": "tw",
  "Tajikistan": "tj",
  "Thailand": "th",
  "Turkmenistan": "tm",
  "UAE": "ae",
  "United-Arab-Emirates": "ae",
  "United Arab Emirates": "ae",
  "Uzbekistan": "uz",
  "Vietnam": "vn",

  // Africa
  "Algeria": "dz",
  "Angola": "ao",
  "Benin": "bj",
  "Botswana": "bw",
  "Burkina-Faso": "bf",
  "Burkina Faso": "bf",
  "Burundi": "bi",
  "Cameroon": "cm",
  "Cape-Verde": "cv",
  "Cape Verde": "cv",
  "Central-African-Republic": "cf",
  "Chad": "td",
  "Comoros": "km",
  "Congo": "cg",
  "Congo-DR": "cd",
  "Cote-D-Ivoire": "ci",
  "Ivory-Coast": "ci",
  "Ivory Coast": "ci",
  "DR-Congo": "cd",
  "Djibouti": "dj",
  "Egypt": "eg",
  "Equatorial-Guinea": "gq",
  "Eritrea": "er",
  "Eswatini": "sz",
  "Ethiopia": "et",
  "Gabon": "ga",
  "Gambia": "gm",
  "Ghana": "gh",
  "Guinea": "gn",
  "Guinea-Bissau": "gw",
  "Kenya": "ke",
  "Lesotho": "ls",
  "Liberia": "lr",
  "Libya": "ly",
  "Madagascar": "mg",
  "Malawi": "mw",
  "Mali": "ml",
  "Mauritania": "mr",
  "Mauritius": "mu",
  "Morocco": "ma",
  "Mozambique": "mz",
  "Namibia": "na",
  "Niger": "ne",
  "Nigeria": "ng",
  "Rwanda": "rw",
  "Senegal": "sn",
  "Sierra-Leone": "sl",
  "Somalia": "so",
  "South-Africa": "za",
  "South Africa": "za",
  "South-Sudan": "ss",
  "Sudan": "sd",
  "Tanzania": "tz",
  "Togo": "tg",
  "Tunisia": "tn",
  "Uganda": "ug",
  "Zambia": "zm",
  "Zimbabwe": "zw",

  // Oceania
  "Fiji": "fj",
  "New-Zealand": "nz",
  "New Zealand": "nz",
  "Papua-New-Guinea": "pg",

  // International / Special
  "World": "un",
  "Europe": "eu",
  "International": "un",
};

/**
 * Get ISO code for a country name
 * Returns undefined if not found
 */
export function getCountryIsoCode(countryName: string): string | undefined {
  // Direct lookup
  if (COUNTRY_TO_ISO[countryName]) {
    return COUNTRY_TO_ISO[countryName];
  }

  // Try with normalized spacing (replace hyphens with spaces)
  const normalized = countryName.replace(/-/g, " ");
  if (COUNTRY_TO_ISO[normalized]) {
    return COUNTRY_TO_ISO[normalized];
  }

  // Try lowercase match
  const lowerName = countryName.toLowerCase();
  for (const [key, value] of Object.entries(COUNTRY_TO_ISO)) {
    if (key.toLowerCase() === lowerName) {
      return value;
    }
  }

  return undefined;
}

/**
 * Get flag SVG path for a country
 * Uses circle-flags npm package
 */
export function getCountryFlagPath(countryName: string): string | undefined {
  const isoCode = getCountryIsoCode(countryName);
  if (!isoCode) return undefined;

  // Path to circle-flags in node_modules (served via Next.js public)
  // We need to copy these to public/ or use a different approach
  return `/flags/${isoCode}.svg`;
}
