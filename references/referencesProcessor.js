// Configure look and feel of output table:
var referencesDataTables = {
    columns: [],
    data: [],
    paging: false
};

// Set column titles:
for(var columnShortform in referencesDataTableColumns) {
    referencesDataTables.columns.push({
        title: referencesDataTableColumns[columnShortform]
    });
}

function makeHtmlLink(linkTitle, linkUrl, linkId) {
    return '<a href="'+ linkUrl + '">' + linkTitle + '</a>';
}

function makeHtmlPageReference() {
    var pageReferenceId = "";
    for (var i in arguments) {
        pageReferenceId += arguments[i]
        .toLowerCase()
        .replace(/ /gi, "_")
        .replace(/:/gi, "_")
        .replace(/\./gi, "_")
        .replace(/\-/gi, "_")
        .replace(/\(/gi, "_")
        .replace(/\)/gi, "_")
        .replace(/\)/gi, "_")
        .replace(/\//gi, "_")
        .replace(/-/gi, "_")
        .replace(/,/gi, "")
        .replace(/;/gi, "")
        .replace(/&/gi, "")
        + "_";
    }
    pageReferenceId = pageReferenceId.slice(0, -1); // Remove trailing underscore.
    pageReferenceId = pageReferenceId.replace(/_+/gi, "_")
    return '<a id="' + pageReferenceId + '" href="#' + pageReferenceId + '">#</a>';
}

// Set table data rows:
for (var i in references) {
    var reference = references[i];
    for(var i in reference.contents) {
        var referenceDataTablesDataRow = [];
        // Note: the order here must correspond to the order on in `referencesDataTableColumns`.
        referenceDataTablesDataRow.push(makeHtmlPageReference(reference.title, reference.contents[i].section));

        referenceDataTablesDataRow.push(reference.title);
        
        referenceDataTablesDataRow.push(reference.contents[i].section);
        
        referenceDataTablesDataRow.push(reference.contents[i].topic);
        
        referenceDataTablesDataRow.push(reference.contents[i].subtopic);
        
        // Use a default if no note exists.
        referenceDataTablesDataRow.push(reference.contents[i].note != null ? reference.contents[i].note : "N/A");
        
        referenceDataTablesDataRow.push(makeHtmlLink(referencesDataTableColumns.url, reference.url));
        referenceDataTablesDataRow.push(makeHtmlLink(referencesDataTableColumns.cached, reference.cached));

        // Validate all data in referenceDataTablesDataRow are strings:
        for (var j in referenceDataTablesDataRow) {
            if (typeof referenceDataTablesDataRow[j] != typeof '') {
                throw new Error("Invalid data: " + reference);
            }
        }

        // Add the row to the DataTable:
        referencesDataTables.data.push(referenceDataTablesDataRow);
    }
}