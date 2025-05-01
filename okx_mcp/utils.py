# okx_mcp/utils.py
import logging
import csv
import io
import yaml
from typing import Optional, Dict, Any, List, Union

logger = logging.getLogger(__name__)

def _format_to_csv(data: List[Dict[str, Any]], fieldnames: List[str]) -> str:
    """Formats a list of dictionaries into a CSV string."""
    if not data:
        return ""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(data)
    return output.getvalue()

def format_data(data: Union[List[Dict[str, Any]], Dict[str, Any]], format_type: str, headers: Optional[List[str]] = None) -> Any:
    """
    Formats the given data into the specified format.

    Args:
        data: The data to format (list of dictionaries or a single dictionary).
        format_type: The desired format ('json', 'csv', 'md', 'yaml').
        headers: Optional list of headers (required for CSV/MD if data is list of dicts).

    Returns:
        The formatted data (string for text formats, original data for 'json').
        Returns an error string if formatting fails or format is unsupported.
    """
    format_type = format_type.lower()
    logger.debug(f"Formatting data to type: {format_type}")

    if not data:
        logger.warning("Formatting requested for empty data.")
        if format_type == 'json':
            return data # Return empty list/dict as is
        else:
            return "No data to format."

    is_list = isinstance(data, list)

    try:
        if format_type == 'json':
            # For JSON, return the raw Python object (list/dict)
            # MCP handles JSON serialization automatically if needed
            return data
        elif format_type == 'csv':
            if not is_list:
                return "error: CSV format requires a list of dictionaries."
            if not headers:
                if data:
                    headers = list(data[0].keys())
                else:
                    return "error: Cannot determine headers for empty CSV data."
            return _format_to_csv(data, headers) # Use existing helper
        elif format_type == 'yaml':
            return yaml.dump(data, allow_unicode=True, sort_keys=False)
        elif format_type == 'md': # Markdown Table or List
            if is_list:
                # Format as markdown table for list of dicts
                if not data:
                    return "No data for Markdown table."
                
                if not headers:
                    headers = list(data[0].keys())

                # Create header row
                header_line = "| " + " | ".join(headers) + " |"
                separator_line = "| " + " | ".join(['---'] * len(headers)) + " |"

                # Create data rows
                data_lines = []
                for row_dict in data:
                    row_values = [str(row_dict.get(h, '')) for h in headers]
                    data_lines.append("| " + " | ".join(row_values) + " |")

                return "\n".join([header_line, separator_line] + data_lines)
            else:
                # Format as markdown list for dictionary
                md_lines = ["# Data"]
                
                def dict_to_md_list(d, level=0):
                    lines = []
                    indent = "  " * level
                    for key, value in d.items():
                        if isinstance(value, dict):
                            lines.append(f"{indent}- **{key}**:")
                            lines.extend(dict_to_md_list(value, level + 1))
                        elif isinstance(value, list) and value and isinstance(value[0], dict):
                            lines.append(f"{indent}- **{key}**:")
                            for i, item in enumerate(value):
                                lines.append(f"{indent}  - Item {i+1}:")
                                lines.extend(dict_to_md_list(item, level + 2))
                        else:
                            lines.append(f"{indent}- **{key}**: {value}")
                    return lines
                
                md_lines.extend(dict_to_md_list(data))
                return "\n".join(md_lines)
        else:
            logger.error(f"Unsupported format type requested: {format_type}")
            return f"error: Unsupported format type '{format_type}'. Supported formats: json, csv, md, yaml."

    except Exception as e:
        logger.exception(f"Error formatting data to {format_type}: {e}")
        return f"error: Failed to format data as {format_type}: {e}" 